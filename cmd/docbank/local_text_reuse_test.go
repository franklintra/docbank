package main

import (
	"context"
	"encoding/json/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
	"go.kenn.io/docbank/document/openaicompat"
	"go.kenn.io/docbank/internal/api"
	"go.kenn.io/docbank/internal/blob"
	"go.kenn.io/docbank/internal/config"
	"go.kenn.io/docbank/internal/home"
	"go.kenn.io/docbank/internal/processing"
	"go.kenn.io/docbank/internal/store"
)

func TestLocalTextSwitchReusesExistingInputsAndIndexesOnlyNewText(t *testing.T) {
	t.Setenv("DOCBANK_TEST_LOCAL_TEXT_KEY", "synthetic-key")
	var oldCalls, localCalls atomic.Int64
	handler := func(calls *atomic.Int64) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			var body struct {
				Input []string `json:"input"`
				Model string   `json:"model"`
			}
			if !assert.NoError(t, json.UnmarshalRead(r.Body, &body)) {
				return
			}
			assert.Equal(t, "nomic-embed-text", body.Model)
			data := make([]map[string]any, len(body.Input))
			for i, input := range body.Input {
				assert.True(t, strings.HasPrefix(input, "search_document: "))
				vector := make([]float32, 768)
				vector[0] = 1
				data[i] = map[string]any{"object": "embedding", "index": i, "embedding": vector}
			}
			w.Header().Set("Content-Type", "application/json")
			assert.NoError(t, json.MarshalWrite(w, map[string]any{"object": "list", "model": "nomic-embed-text", "data": data}))
		})
	}
	oldServer := httptest.NewServer(handler(&oldCalls))
	defer oldServer.Close()
	localServer := httptest.NewServer(handler(&localCalls))
	defer localServer.Close()
	cfg, err := config.Default().WithLocalText()
	require.NoError(t, err)
	p := cfg.EmbeddingProfiles["local-text"]
	p.Runtime.Managed = false
	p.Runtime.ProviderEgressConfig = config.ProviderEgressConfig{Endpoint: oldServer.URL, AllowedCIDRs: []string{"127.0.0.0/8"},
		ProxyMode: "disabled", ConnectTimeout: config.Duration(time.Second), KeepAlive: config.Duration(time.Second), TLSHandshakeTimeout: config.Duration(time.Second)}
	cfg.CredentialBindings["local-text"] = config.CredentialBindingConfig{EnvironmentVariable: "DOCBANK_TEST_LOCAL_TEXT_KEY"}
	input, err := cfg.EmbeddingModelInput("local-text")
	require.NoError(t, err)
	descriptor := config.EmbeddingDescriptor(p, input)
	wire := openaicompat.Profile{Origin: oldServer.URL, Descriptor: descriptor, ModelInput: input, SecretBinding: p.CredentialBinding,
		DeploymentEpoch: p.Runtime.DeploymentEpoch, RequestTimeout: p.Runtime.RequestTimeout.Std(), MaxBatchItems: p.MaxBatchItems,
		MaxInputBytes: p.MaxInputBytes, MaxRequestBytes: p.Runtime.MaxRequestBytes, MaxResponseBytes: p.MaxResponseBytes,
		EgressPolicy: providerEgressPolicy(p.Runtime.ProviderEgressConfig)}
	descriptor, _, err = finalizeOpenAIEmbeddingDescriptor(wire)
	require.NoError(t, err)
	p.DescriptorFingerprint = descriptor.Fingerprint
	cfg.EmbeddingProfiles["semantic"] = p
	delete(cfg.EmbeddingProfiles, "local-text")
	archive := cfg.ProcessingProfiles["local-text"]
	archive.Embeddings = []string{"semantic"}
	cfg.ProcessingProfiles["archive"] = archive
	delete(cfg.ProcessingProfiles, "local-text")
	shared := archive
	shared.ConsentFingerprint = sha256Hex("shared processing consent")
	cfg.ProcessingProfiles["shared"] = shared
	layout := home.Layout{Root: t.TempDir()}
	require.NoError(t, layout.Ensure())
	catalog, err := store.Open(layout.DBPath())
	require.NoError(t, err)
	defer func() { require.NoError(t, catalog.Close()) }()
	blobs, err := blob.New(store.NewPackCatalog(catalog), layout.BlobsDir())
	require.NoError(t, err)
	defer func() { require.NoError(t, blobs.Close()) }()
	bundle, err := configureEmbeddingRuntimeBundle(cfg, blobs, layout.BlobTmpDir())
	require.NoError(t, err)
	profiles, err := executableProcessingProfiles(cfg, bundle)
	require.NoError(t, err)
	service, err := processing.NewService(processing.ServiceConfig{Catalog: catalog, Blobs: blobs, Gate: api.NewOperationGate(), Profiles: profiles, SpoolDirectory: layout.BlobTmpDir()})
	require.NoError(t, err)
	receipt, err := blobs.WriteDetailedContext(t.Context(), strings.NewReader("old synthetic passage"))
	require.NoError(t, err)
	encoding, err := receipt.EncodingName()
	require.NoError(t, err)
	first, err := catalog.CreateFile(t.Context(), catalog.RootID(), "first.txt", receipt.Hash, receipt.Size, "text/plain", store.BlobPhysical{Encoding: encoding, StoredBytes: receipt.StoredSize})
	require.NoError(t, err)
	selector := processing.Selector{Profile: "archive", NodeID: first.ID, ContentVersionID: first.CurrentVersionID}
	plan, err := service.Plan(t.Context(), selector)
	require.NoError(t, err)
	_, err = service.Start(t.Context(), processing.StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint, Consent: true})
	require.NoError(t, err)
	require.Equal(t, int64(1), oldCalls.Load())
	require.NoError(t, service.Shutdown(t.Context()))

	auto, managed, _, err := localTextCandidate(cfg, "archive", "auto")
	require.NoError(t, err)
	require.False(t, managed)
	require.Equal(t, oldServer.URL, auto.EmbeddingProfiles["semantic"].Runtime.Endpoint)
	configPath := filepath.Join(layout.Root, "config.toml")
	require.NoError(t, os.WriteFile(configPath, []byte("# keep this comment\n"), 0600))
	require.NoError(t, saveLocalText(layout.Root, config.LocalTextConfig{Profile: "archive", Server: "local"}))
	choice, err := readLocalText(layout.Root)
	require.NoError(t, err)
	candidate, managed, rebuild, err := localTextCandidate(cfg, choice.Profile, choice.Server)
	require.NoError(t, err)
	require.True(t, managed)
	require.False(t, rebuild)
	content, err := os.ReadFile(configPath)
	require.NoError(t, err)
	require.Equal(t, "# keep this comment\n", string(content))
	bundle, err = configureEmbeddingRuntimeBundle(candidate, blobs, layout.BlobTmpDir())
	require.NoError(t, err)
	managedConfig, err := candidate.EmbeddingProfile("semantic")
	require.NoError(t, err)
	managedProfile, err := config.ManagedEmbeddingProfile(managedConfig, func(context.Context) (string, error) { return localServer.URL, nil })
	require.NoError(t, err)
	managedProvider, err := openaicompat.New(managedProfile, nil, &http.Client{})
	require.NoError(t, err)
	for name, provider := range bundle.providers {
		if provider.Descriptor().Fingerprint == managedProfile.Descriptor.Fingerprint {
			bundle.providers[name] = managedProvider
		}
	}

	selected, err := executableProcessingProfiles(candidate, bundle)
	require.NoError(t, err)
	effective, err := projectLocalTextProfile(cfg, profiles, candidate, "archive", selected["archive"])
	require.NoError(t, err)
	require.Equal(t, oldServer.URL, cfg.EmbeddingProfiles["semantic"].Runtime.Endpoint)
	service, err = processing.NewService(processing.ServiceConfig{Catalog: catalog, Blobs: blobs, Gate: api.NewOperationGate(), Profiles: profiles, SpoolDirectory: layout.BlobTmpDir()})
	require.NoError(t, err)
	defer func() { require.NoError(t, service.Shutdown(t.Context())) }()
	resolved, err := effective.ProcessingProfile("archive")
	require.NoError(t, err)
	_, fingerprints, err := document.CanonicalProfile(resolved.Document)
	require.NoError(t, err)
	plan, err = service.Plan(t.Context(), selector)
	require.NoError(t, err)
	_, err = service.Start(t.Context(), processing.StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint, Consent: true})
	require.NoError(t, err)
	require.Equal(t, int64(1), oldCalls.Load())
	require.Zero(t, localCalls.Load(), "switching endpoints must reuse existing vectors")
	backfillCtx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	setup := &processing.LocalTextSetup{Service: service, Selected: "archive"}
	go func() { done <- setup.Backfill(backfillCtx, nil) }()
	stopped := false
	t.Cleanup(func() {
		if !stopped {
			cancel()
			<-done
		}
	})
	receipt, err = blobs.WriteDetailedContext(t.Context(), strings.NewReader("new synthetic passage"))
	require.NoError(t, err)
	encoding, err = receipt.EncodingName()
	require.NoError(t, err)
	second, err := catalog.CreateFile(t.Context(), catalog.RootID(), "second.txt", receipt.Hash, receipt.Size, "text/plain", store.BlobPhysical{Encoding: encoding, StoredBytes: receipt.StoredSize})
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		jobs, err := catalog.EmbeddingJobsForVersionProfile(t.Context(), second.CurrentVersionID, fingerprints.Profile)
		return err == nil && len(jobs) == 1 && jobs[0].State == "completed"
	}, 10*time.Second, time.Millisecond)
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	stopped = true
	require.Equal(t, int64(1), oldCalls.Load())
	require.Equal(t, int64(1), localCalls.Load())
	missing, err := catalog.MissingProcessingTargetsAfter(t.Context(), fingerprints.Profile, "", 1, 50)
	require.NoError(t, err)
	require.Empty(t, missing, "completed versions stay excluded after service restart")
	require.NoError(t, service.Shutdown(t.Context()))
	service, err = processing.NewService(processing.ServiceConfig{Catalog: catalog, Blobs: blobs, Gate: api.NewOperationGate(), Profiles: profiles, SpoolDirectory: layout.BlobTmpDir()})
	require.NoError(t, err)
	missing, err = catalog.MissingProcessingTargetsAfter(t.Context(), fingerprints.Profile, "", 1, 50)
	require.NoError(t, err)
	require.Empty(t, missing)
	receipt, err = blobs.WriteDetailedContext(t.Context(), strings.NewReader("third distinct synthetic passage"))
	require.NoError(t, err)
	encoding, err = receipt.EncodingName()
	require.NoError(t, err)
	third, err := catalog.CreateFile(t.Context(), catalog.RootID(), "third.txt", receipt.Hash, receipt.Size, "text/plain", store.BlobPhysical{Encoding: encoding, StoredBytes: receipt.StoredSize})
	require.NoError(t, err)
	_, err = service.RevokeConsent(t.Context())
	require.NoError(t, err)
	selector = processing.Selector{Profile: "archive", NodeID: third.ID, ContentVersionID: third.CurrentVersionID}
	plan, err = service.Plan(t.Context(), selector)
	require.NoError(t, err)
	_, err = service.Start(t.Context(), processing.StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint})
	require.ErrorIs(t, err, processing.ErrConsentRequired)
	selector.Profile = "shared"
	plan, err = service.Plan(t.Context(), selector)
	require.NoError(t, err)
	_, err = service.Start(t.Context(), processing.StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint, Consent: true})
	require.NoError(t, err)
	require.Equal(t, int64(2), oldCalls.Load(), "another profile sharing the definition retains its provider")
	require.Equal(t, int64(1), localCalls.Load())
}

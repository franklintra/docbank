package main

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document/openaicompat"
	"go.kenn.io/docbank/internal/config"
	"go.kenn.io/docbank/internal/processing"
	"go.kenn.io/kit/embedlocal"
)

func TestLocalTextApproval(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, answer, mode   string
		ready                bool
		missingBytes         int64
		downloaded           bool
		wantError, bytesLine string
	}{
		{"installed_decline", "n\n", "local", false, 0, false, "declined", "missing download: 0 bytes"},
		{"installed_approve", "y\n", "local", false, 0, false, "", "missing download: 0 bytes"},
		{"missing_decline", "n\n", "local", false, 295000000, false, "declined", "missing download: 295000000 bytes"},
		{"missing_approve", "y\n", "local", false, 295000000, true, "", "missing download: 295000000 bytes"},
		{"auto_ollama", "y\n", "auto", true, 0, false, "", "missing download: 0 bytes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &cobra.Command{}
			cmd.SetContext(t.Context())
			cmd.SetIn(bytes.NewBufferString(tc.answer))
			output := new(bytes.Buffer)
			cmd.SetOut(output)
			downloaded := false
			err := approveLocalTextWithRuntime(cmd, true, tc.mode, func(_ context.Context, opts embedlocal.Options) (string, error) {
				require.Equal(t, tc.mode == "local", opts.SkipOllama)

				if tc.ready {
					return "http://127.0.0.1:11434/v1", nil
				}
				return "", embedlocal.ErrNotFound
			}, func(ctx context.Context, opts embedlocal.InstallOptions) (embedlocal.Installation, error) {
				require.False(t, tc.ready, "a ready server must not trigger installation")
				if tc.missingBytes > 0 {
					approved, err := opts.Confirm(ctx, tc.missingBytes)
					if err != nil {
						return embedlocal.Installation{}, fmt.Errorf("approve installation: %w", err)
					}
					require.True(t, approved)
					require.Contains(t, output.String(), "[y/N]")
					downloaded = true
				}
				return embedlocal.Installation{Executable: "synthetic-kennembed", DataDir: t.TempDir()}, nil
			})
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.downloaded, downloaded)
			require.Contains(t, output.String(), tc.bytesLine)
			require.Equal(t, 1, bytes.Count(output.Bytes(), []byte("[y/N]")))
		})
	}
}

func TestLocalTextCandidatePreservesOtherProvidersAndValidatesManagedModel(t *testing.T) {
	t.Parallel()
	cfg, _ := embeddingGemma2Config(t, "http://127.0.0.1:11434", "DOCBANK_TEST_SWITCH_KEY")
	render := newReportRenditionFixture(t)
	cfg.RenditionProfiles = render.config.RenditionProfiles
	cfg.RetrievalProfiles = render.config.RetrievalProfiles
	cfg.ProcessingProfiles = render.config.ProcessingProfiles
	p := cfg.ProcessingProfiles["archive"]
	p.Embeddings = []string{"semantic"}
	cfg.ProcessingProfiles["archive"] = p
	before, err := cfg.ProcessingProfile("archive")
	require.NoError(t, err)
	candidate, managed, rebuild, err := localTextCandidate(cfg, "archive", "local")
	require.NoError(t, err)
	require.True(t, managed)
	require.True(t, rebuild)
	after, err := candidate.ProcessingProfile("archive")
	require.NoError(t, err)
	require.Equal(t, before.Document.Rendition, after.Document.Rendition, "PDF renderer and its disclosure survive the model switch")
	require.Equal(t, before.Retrieval, after.Retrieval)
	require.Equal(t, "semantic", after.Document.Embeddings[0].Name)
	require.Equal(t, "nomic-embed-text", after.Document.Embeddings[0].Model)
	require.Equal(t, "embeddinggemma2-text-f32-768-v1", before.Document.Embeddings[0].Model)
	for _, mutate := range []func(*config.EmbeddingProfileConfig){
		func(p *config.EmbeddingProfileConfig) { p.Model = "unsupported" },
		func(p *config.EmbeddingProfileConfig) { p.Dimensions = 512 },
	} {
		e := candidate.EmbeddingProfiles["semantic"]
		mutate(&e)
		_, err := config.ManagedEmbeddingProfile(e, nil)
		require.ErrorContains(t, err, "nomic-embed-text with 768 dimensions")
	}
	cfg, err = config.Default().WithLocalText()
	require.NoError(t, err)
	e := cfg.EmbeddingProfiles[config.LocalTextProfile]
	e.Dimensions = 512
	e.Runtime.Managed = false
	cfg.EmbeddingProfiles[config.LocalTextProfile] = e
	_, managed, rebuild, err = localTextCandidate(cfg, config.LocalTextProfile, "local")
	require.False(t, managed)
	require.False(t, rebuild)
	require.ErrorIs(t, err, processing.ErrLocalTextConfiguration)
	require.ErrorContains(t, err, "configure 768 dimensions")
}

func TestLocalTextSelectionFormat(t *testing.T) {
	t.Parallel()
	for _, data := range []string{`{"version":1`, `{"version":1,"profile":"archive","server":"invalid"}`, `{"profile":"archive","server":"local"}`, `{"version":2,"profile":"archive","server":"local"}`, `{"version":1,"profile":"archive","server":"local","typo":true}`} {
		t.Run(data, func(t *testing.T) {
			root := t.TempDir()
			require.NoError(t, os.WriteFile(filepath.Join(root, "local-text.json"), []byte(data), 0600))
			_, err := readLocalText(root)
			require.Error(t, err)
		})
	}
	root := t.TempDir()
	require.NoError(t, saveLocalText(root, config.LocalTextConfig{Profile: "archive", Server: "local"}))
	choice, err := readLocalText(root)
	require.NoError(t, err)
	require.Equal(t, "archive", choice.Profile)
}

func TestLocalTextDefaultSelectsNomic(t *testing.T) {
	t.Parallel()
	cfg, err := config.Default().WithLocalText()
	require.NoError(t, err)
	paid, _ := embeddingGemma2Config(t, "http://127.0.0.1:11434", "DOCBANK_TEST_DEFAULT_KEY")
	cfg.EmbeddingProfiles["paid"] = paid.EmbeddingProfiles["semantic"]
	cfg.CredentialBindings = paid.CredentialBindings
	p := cfg.ProcessingProfiles[config.LocalTextProfile]
	p.Embeddings = []string{"paid"}
	cfg.ProcessingProfiles["aaa-paid"] = p
	require.Equal(t, "local-text", defaultLocalTextProfile(cfg))
	cfg.ProcessingProfiles["zzz-nomic"] = cfg.ProcessingProfiles[config.LocalTextProfile]
	delete(cfg.ProcessingProfiles, config.LocalTextProfile)
	require.Equal(t, "zzz-nomic", defaultLocalTextProfile(cfg))
	delete(cfg.ProcessingProfiles, "zzz-nomic")
	require.Equal(t, "local-text", defaultLocalTextProfile(cfg))
	p.Embeddings = []string{config.LocalTextProfile}
	cfg.ProcessingProfiles["zzz-nomic"] = p
	missing := cfg.EmbeddingProfiles[config.LocalTextProfile]
	missing.Runtime = nil
	cfg.EmbeddingProfiles[config.LocalTextProfile] = missing
	require.Equal(t, "local-text", defaultLocalTextProfile(cfg))
}

func TestLocalTextStaleSelectionLeavesDaemonAvailable(t *testing.T) {
	for _, scenario := range []string{"missing_profile", "malformed", "newer", "dimensions"} {
		t.Run(scenario, func(t *testing.T) {
			root := t.TempDir()
			t.Setenv("DOCBANK_HOME", root)
			t.Setenv("DO_NOT_TRACK", "1")
			data := []byte(`{"version":1,"profile":"deleted-profile","server":"local"}`)
			switch scenario {
			case "malformed":
				data = []byte(`{"version":1`)
			case "newer":
				data = []byte(`{"version":2,"profile":"local-text","server":"local"}`)
			case "dimensions":
				cfg, err := config.Default().WithLocalText()
				require.NoError(t, err)
				e := cfg.EmbeddingProfiles[config.LocalTextProfile]
				e.Dimensions = 512
				e.Runtime.Managed = false
				e.Runtime.ProviderEgressConfig = config.ProviderEgressConfig{Endpoint: "http://127.0.0.1:11434", AllowedCIDRs: []string{"127.0.0.0/8"},
					ProxyMode: "disabled", ConnectTimeout: config.Duration(time.Second), KeepAlive: config.Duration(time.Second), TLSHandshakeTimeout: config.Duration(time.Second)}
				input, err := cfg.EmbeddingModelInput(config.LocalTextProfile)
				require.NoError(t, err)
				descriptor, _, err := finalizeOpenAIEmbeddingDescriptor(openaicompat.Profile{Origin: e.Runtime.Endpoint, Descriptor: config.EmbeddingDescriptor(e, input),
					ModelInput: input, SecretBinding: e.CredentialBinding, DeploymentEpoch: e.Runtime.DeploymentEpoch,
					RequestTimeout: e.Runtime.RequestTimeout.Std(), MaxBatchItems: e.MaxBatchItems, MaxInputBytes: e.MaxInputBytes,
					MaxRequestBytes: e.Runtime.MaxRequestBytes, MaxResponseBytes: e.MaxResponseBytes, EgressPolicy: providerEgressPolicy(e.Runtime.ProviderEgressConfig)})
				require.NoError(t, err)
				e.DescriptorFingerprint = descriptor.Fingerprint
				e.Runtime = nil
				cfg.EmbeddingProfiles[config.LocalTextProfile] = e
				cfg.CredentialBindings[config.LocalTextProfile] = config.CredentialBindingConfig{EnvironmentVariable: "DOCBANK_TEST_RESTORE_KEY"}
				var encoded bytes.Buffer
				require.NoError(t, toml.NewEncoder(&encoded).Encode(map[string]any{
					"embedding_profiles": cfg.EmbeddingProfiles, "rendition_profiles": cfg.RenditionProfiles,
					"retrieval_profiles": cfg.RetrievalProfiles, "processing_profiles": cfg.ProcessingProfiles,
					"credential_bindings": cfg.CredentialBindings,
				}))
				encoded.WriteString(`[embedding_profiles.local-text.runtime]
endpoint = "http://127.0.0.1:11434"
allowed_cidrs = ["127.0.0.0/8"]
proxy_mode = "disabled"
connect_timeout = "1s"
keep_alive = "1s"
tls_handshake_timeout = "1s"
adapter_contract = "docbank-openai-compatible-embeddings/v1"
model_revision = "nomic-embed-text-v1.5"
deployment_epoch = "nomic-embed-text-v1.5"
request_timeout = "30s"
max_request_bytes = 2097152
`)
				require.NoError(t, os.WriteFile(filepath.Join(root, "config.toml"), encoded.Bytes(), 0600))
				loaded, err := config.Load(root)
				require.NoError(t, err)
				require.NoError(t, loaded.Validate())
				data = []byte(`{"version":1,"profile":"local-text","server":"local"}`)
			}
			require.NoError(t, os.WriteFile(filepath.Join(root, "local-text.json"), data, 0600))
			stop := startServe(t)
			defer stop()
			record := waitForDaemon(t, root)
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+record.Address+"/api/v1/jobs", nil)
			require.NoError(t, err)
			req.Header.Set("X-Api-Key", record.Metadata["api_key"])
			response, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer func() { require.NoError(t, response.Body.Close()) }()
			var jobs bytes.Buffer
			_, err = jobs.ReadFrom(response.Body)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, response.StatusCode, jobs.String())
			require.NotContains(t, jobs.String(), "embed:local-text")
			saved, err := os.ReadFile(filepath.Join(root, "local-text.json"))
			require.NoError(t, err)
			require.Equal(t, data, saved)
		})
	}
}

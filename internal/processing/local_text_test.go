package processing

import (
	"bytes"
	"context"
	"encoding/json/v2"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
	"go.kenn.io/docbank/document/media/mediatest"
	"go.kenn.io/docbank/document/plaintext"
	"go.kenn.io/docbank/internal/retrieval"
	"go.kenn.io/docbank/internal/store"
)

func TestLocalTextFailedConsentAndSavePreserveSelection(t *testing.T) {
	t.Parallel()
	fixture := newPublicationFixture(t)
	plain, err := plaintext.New(plaintext.Profile{MaxDocumentBytes: 1 << 20})
	require.NoError(t, err)
	renderer := plain
	stored := workerProcessingProfile(t, renderer.Descriptor())
	var portable document.ProcessingProfileV1
	require.NoError(t, json.Unmarshal(stored.CanonicalProfile, &portable))
	portable.Rendition.TrustBoundary = string(renderer.Descriptor().TrustBoundary)
	config := ServiceConfig{Catalog: fixture.catalog, Blobs: fixture.blobs, Gate: newWorkerTestGate(), SpoolDirectory: t.TempDir(),
		Profiles: map[string]ProfileConfig{"text": {Profile: portable, RenditionProvider: renderer,
			RerankingProvider:   consumerReranker{source: &consumerSource{}},
			RerankingProfile:    retrieval.RerankingProfile{ID: "synthetic-reranker", MaxCandidates: 2, MaxExcerptBytes: 4096},
			RerankingDeadline:   time.Second,
			RerankingDisclosure: RuntimeDisclosure{ImmediateProcessor: "synthetic", UltimateProcessor: "synthetic", Endpoint: "https://rerank.example.invalid", Deployment: workerHash("reranker")},
		}}}
	service, err := NewService(config)
	require.NoError(t, err)
	defer service.Stop()
	setup := &LocalTextSetup{Service: service, Selected: "text", DefaultProfile: "text"}
	gateFailure := errors.New("synthetic grant failure")
	saveFailure := errors.New("synthetic save failure")
	setup.Candidate = func(string, string) (*Service, bool, bool, error) {
		next := config
		next.Gate = &localTextGrantGate{workerTestGate: newWorkerTestGate(), err: gateFailure}
		s, err := NewService(next)
		return s, false, false, err
	}
	setup.Startup = func(context.Context, bool, string) error { return nil }
	saved := "previous-profile"
	setup.Save = func(name, _ string) error { saved = name; return nil }
	// Preview and apply review the operation without reopening originals.
	plan, err := setup.Plan(t.Context(), "text", "local")
	require.NoError(t, err)
	require.Len(t, plan.Flow, 2)
	require.Equal(t, "reranking", plan.Flow[1].Capability)
	_, err = setup.Apply(t.Context(), "text", "local", plan.Fingerprint)
	require.ErrorIs(t, err, gateFailure)
	require.Equal(t, "previous-profile", saved)
	setup.Candidate = func(string, string) (*Service, bool, bool, error) {
		s, err := NewService(config)
		return s, false, false, err
	}
	setup.Save = func(string, string) error { return saveFailure }
	_, err = setup.Apply(t.Context(), "text", "local", plan.Fingerprint)
	require.ErrorIs(t, err, ErrConsentSaved)
	require.ErrorIs(t, err, saveFailure)
	require.Equal(t, "previous-profile", saved)
	for _, request := range service.profileConsentRequests(service.profiles["text"]) {
		_, err := fixture.catalog.AuthorizeProviderOperation(t.Context(), request)
		require.NoError(t, err, "consent remains saved when the selection write fails")
	}
}

type localTextGrantGate struct {
	*workerTestGate

	err error
}

func (gate *localTextGrantGate) MutateContext(context.Context, func() error) error { return gate.err }

func TestLocalTextRetriesAfterPDFPasswordBecomesAvailable(t *testing.T) {
	t.Parallel()
	fixture := newPublicationFixture(t)
	renderer := newWorkerProvider(t)
	provider := &passwordCountingProvider{RenditionProvider: renderer, supported: true}
	stored := workerProcessingProfile(t, provider.Descriptor())
	var portable document.ProcessingProfileV1
	require.NoError(t, json.Unmarshal(stored.CanonicalProfile, &portable))
	portable.Rendition.TrustBoundary = string(provider.Descriptor().TrustBoundary)
	service, err := NewService(ServiceConfig{Catalog: fixture.catalog, Blobs: fixture.blobs, Gate: newWorkerTestGate(), SpoolDirectory: t.TempDir(),
		Profiles: map[string]ProfileConfig{"pdf": {Profile: portable, RenditionProvider: provider}}})
	require.NoError(t, err)
	defer service.Stop()
	encrypted, err := mediatest.EncryptedPDF("open", "owner", 128)
	require.NoError(t, err)
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(encrypted))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "locked.pdf", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	require.NoError(t, service.grantProfileConsent(t.Context(), service.profiles["pdf"], nil))
	setup := &LocalTextSetup{Service: service, Selected: "pdf"}
	item := store.DocumentSummary{NodeID: node.ID, ContentVersionID: node.CurrentVersionID}
	require.NoError(t, setup.process(t.Context(), item))
	require.Zero(t, renderer.calls)
	provider.passwords = []string{"open"}
	require.NoError(t, setup.process(t.Context(), item))
	require.Equal(t, 1, renderer.calls, "the same version must be reconsidered after the password source changes")
}

func TestLocalTextReconsidersRenamedPDF(t *testing.T) {
	t.Parallel()
	fixture := newPublicationFixture(t)
	renderer := newWorkerProvider(t)
	stored := workerProcessingProfile(t, renderer.Descriptor())
	var portable document.ProcessingProfileV1
	require.NoError(t, json.Unmarshal(stored.CanonicalProfile, &portable))
	portable.Rendition.DiscloseFilename = true
	portable.Rendition.TrustBoundary = string(renderer.Descriptor().TrustBoundary)
	service, err := NewService(ServiceConfig{Catalog: fixture.catalog, Blobs: fixture.blobs, Gate: newWorkerTestGate(), SpoolDirectory: t.TempDir(),
		Profiles: map[string]ProfileConfig{"pdf": {Profile: portable, RenditionProvider: renderer}}})
	require.NoError(t, err)
	defer service.Stop()
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(mediatest.PDF()))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "report.txt", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	require.NoError(t, service.grantProfileConsent(t.Context(), service.profiles["pdf"], nil))
	setup := &LocalTextSetup{Service: service, Selected: "pdf"}
	item := store.DocumentSummary{NodeID: node.ID, ContentVersionID: node.CurrentVersionID}
	require.NoError(t, setup.process(t.Context(), item))
	require.Zero(t, renderer.calls)
	renamed, _, err := fixture.catalog.Move(t.Context(), node.ID, *node.ParentID, "report.pdf", store.UnconditionalRev)
	require.NoError(t, err)
	require.Equal(t, item.ContentVersionID, renamed.CurrentVersionID)
	require.NoError(t, setup.process(t.Context(), item))
	require.Equal(t, 1, renderer.calls)
}

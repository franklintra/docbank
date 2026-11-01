package processing

import (
	"context"
	"encoding/json/v2"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
	"go.kenn.io/docbank/document/plaintext"
	"go.kenn.io/docbank/internal/retrieval"
	"go.kenn.io/docbank/internal/store"
)

type localTextRenditionSpy struct {
	document.RenditionProvider

	descriptors atomic.Int64
}

func (p *localTextRenditionSpy) Descriptor() document.RenditionDescriptor {
	p.descriptors.Add(1)
	return p.RenditionProvider.Descriptor()
}

func TestLocalTextIneligibleCacheAndConsentRollback(t *testing.T) {
	t.Parallel()
	fixture := newPublicationFixture(t)
	plain, err := plaintext.New(plaintext.Profile{MaxDocumentBytes: 1 << 20})
	require.NoError(t, err)
	renderer := &localTextRenditionSpy{RenditionProvider: plain}
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
	node, err := fixture.catalog.ContentVersionByID(t.Context(), fixture.versionID)
	require.NoError(t, err)
	setup := &LocalTextSetup{Service: service, Selected: "text", DefaultProfile: "text"}
	target := store.DocumentSummary{NodeID: node.NodeID, ContentVersionID: node.ID}
	before := renderer.descriptors.Load()
	require.NoError(t, setup.process(t.Context(), target))
	inspected := renderer.descriptors.Load()
	require.Greater(t, inspected, before, "source inspection asks the renderer for its policy descriptor")
	require.NoError(t, fixture.blobs.Remove(node.BlobHash))
	require.NoError(t, setup.process(t.Context(), target), "cached ineligible versions need no further source access")
	require.Equal(t, inspected, renderer.descriptors.Load())
	gateFailure := errors.New("synthetic grant failure")
	cleanupFailure := errors.New("synthetic cleanup failure")
	setup.Candidate = func(string, string) (*Service, bool, bool, error) {
		next := config
		next.Gate = &localTextGrantGate{workerTestGate: newWorkerTestGate(), err: gateFailure}
		s, err := NewService(next)
		return s, false, false, err
	}
	setup.Startup = func(context.Context, bool, string) error { return nil }
	saved := false
	setup.Save = func(string, string) error { saved = true; return nil }
	setup.Remove = func() error { saved = false; return nil }
	// Apply reviews the operation without reopening originals.
	plan, reviewed, err := setup.review(t.Context(), "text", "local", false)
	require.NoError(t, err)
	reviewed.Stop()
	require.Len(t, plan.Flow, 2)
	require.Equal(t, "reranking", plan.Flow[1].Capability)
	_, err = setup.Apply(t.Context(), "text", "local", plan.Fingerprint)
	require.ErrorIs(t, err, gateFailure)
	require.False(t, saved)
	setup.Remove = func() error { return cleanupFailure }
	_, err = setup.Apply(t.Context(), "text", "local", plan.Fingerprint)
	require.ErrorIs(t, err, gateFailure)
	require.ErrorIs(t, err, cleanupFailure)
	require.ErrorContains(t, err, "partial state")
	require.True(t, saved)
}

type localTextGrantGate struct {
	*workerTestGate

	err error
}

func (gate *localTextGrantGate) MutateContext(context.Context, func() error) error { return gate.err }

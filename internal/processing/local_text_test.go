package processing

import (
	"bytes"
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
	"go.kenn.io/docbank/document/media/mediatest"
	"go.kenn.io/docbank/document/plaintext"
	"go.kenn.io/docbank/internal/retrieval"
	"go.kenn.io/docbank/internal/store"
	"go.kenn.io/docbank/internal/vectorworker"
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
	renderer := newWorkerProvider(t)
	provider := &passwordCountingProvider{RenditionProvider: renderer, supported: true}
	fixture, setup := newLocalTextBackfillFixture(t, provider)
	targets, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Len(t, targets, 1)
	require.NoError(t, setup.process(t.Context(), targets[0]))
	require.Zero(t, provider.calls, "ordinary ineligibility does not read the password list")
	encrypted, err := mediatest.EncryptedPDF("open", "owner", 128)
	require.NoError(t, err)
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(encrypted))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "locked.pdf", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	item := store.DocumentSummary{NodeID: node.ID, ContentVersionID: node.CurrentVersionID, Name: node.Name}
	require.NoError(t, setup.process(t.Context(), item))
	require.Zero(t, renderer.calls)
	targets, err = setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	for _, target := range targets {
		require.NotEqual(t, item.ContentVersionID, target.ContentVersionID)
	}
	provider.passwords = []string{"open"}
	targets, err = setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Contains(t, targets, item)
	require.NoError(t, setup.process(t.Context(), item))
	require.Equal(t, 1, renderer.calls, "the same version must be reconsidered after the password source changes")
}

func TestLocalTextReconsidersRenamedPDF(t *testing.T) {
	t.Parallel()
	renderer := newWorkerProvider(t)
	fixture, setup := newLocalTextBackfillFixture(t, renderer)
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(mediatest.PDF()))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "report.txt", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	item := store.DocumentSummary{NodeID: node.ID, ContentVersionID: node.CurrentVersionID, Name: node.Name}
	require.NoError(t, setup.process(t.Context(), item))
	require.Zero(t, renderer.calls)
	renamed, _, err := fixture.catalog.Move(t.Context(), node.ID, *node.ParentID, "report.pdf", store.UnconditionalRev)
	require.NoError(t, err)
	require.Equal(t, item.ContentVersionID, renamed.CurrentVersionID)
	item.Name = renamed.Name
	targets, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Contains(t, targets, item)
	require.NoError(t, setup.process(t.Context(), item))
	require.Equal(t, 1, renderer.calls)
}

func newLocalTextBackfillFixture(tb testing.TB, renderer document.RenditionProvider, providers ...document.EmbeddingProvider) (publicationFixture, *LocalTextSetup) {
	tb.Helper()
	fixture := newPublicationFixture(tb)
	stored := workerProcessingProfile(tb, renderer.Descriptor())
	var portable, embeddings document.ProcessingProfileV1
	require.NoError(tb, json.Unmarshal(stored.CanonicalProfile, &portable))
	embedder := document.EmbeddingProvider(localTextEmbeddingProvider{descriptor: embeddingWorkerDescriptor(tb)})
	if len(providers) != 0 {
		embedder = providers[0]
	}
	descriptor := embedder.Descriptor()
	require.NoError(tb, json.Unmarshal(embeddingWorkerProfile(tb, descriptor).CanonicalProfile, &embeddings))
	portable.Embeddings = embeddings.Embeddings[:1]
	portable.Rendition.DiscloseFilename = true
	portable.Rendition.TrustBoundary = string(renderer.Descriptor().TrustBoundary)
	service, err := NewService(ServiceConfig{Catalog: fixture.catalog, Blobs: fixture.blobs, Gate: newWorkerTestGate(), SpoolDirectory: tb.TempDir(),
		Profiles: map[string]ProfileConfig{"pdf": {Profile: portable, RenditionProvider: renderer,
			EmbeddingProviders: map[string]document.EmbeddingProvider{"chunk": embedder},
			Tokenizers:         map[string]document.Tokenizer{"chunk": embeddingIntegrationTokenizer{}},
		}}})
	require.NoError(tb, err)
	tb.Cleanup(service.Stop)
	require.NoError(tb, service.grantProfileConsent(tb.Context(), service.profiles["pdf"], nil))
	return fixture, &LocalTextSetup{Service: service, Selected: "pdf"}
}

func TestLocalTextSkipsIneligibleSweepAndPrunesAbsentInputs(t *testing.T) {
	t.Parallel()
	fixture, setup := newLocalTextBackfillFixture(t, newWorkerProvider(t))
	version, err := fixture.catalog.ContentVersionByID(t.Context(), fixture.versionID)
	require.NoError(t, err)
	for i := range 1029 {
		_, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), fmt.Sprintf("ineligible-%04d.pdf", i), version.BlobHash, version.Size, "application/pdf")
		require.NoError(t, err)
	}
	count, after := 0, ""
	for {
		targets, err := setup.targets(t.Context(), after, 50)
		require.NoError(t, err)
		if len(targets) == 0 {
			break
		}
		for _, target := range targets {
			require.NoError(t, setup.process(t.Context(), target))
		}
		count += len(targets)
		after = targets[len(targets)-1].ContentVersionID
	}
	require.Equal(t, 1030, count)
	targets, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Empty(t, targets, "a fully ineligible sweep lets the lane idle")
	node, err := fixture.catalog.NodeByID(t.Context(), version.NodeID)
	require.NoError(t, err)
	trashed, _, err := fixture.catalog.Trash(t.Context(), node.ID, node.Revision)
	require.NoError(t, err)
	targets, err = setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Empty(t, targets)
	_, _, err = fixture.catalog.Restore(t.Context(), node.ID, trashed.Revision)
	require.NoError(t, err)
	targets, err = setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Equal(t, []store.DocumentSummary{{NodeID: node.ID, ContentVersionID: version.ID, Name: node.Name}}, targets)
}

func TestLocalTextInspectionErrorsRemainRetryable(t *testing.T) {
	t.Parallel()
	fixture, setup := newLocalTextBackfillFixture(t, newWorkerProvider(t))
	targets, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Len(t, targets, 1)
	version, err := fixture.catalog.ContentVersionByID(t.Context(), fixture.versionID)
	require.NoError(t, err)
	require.NoError(t, fixture.blobs.Remove(version.BlobHash))
	require.Error(t, setup.process(t.Context(), targets[0]))
	retry, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Equal(t, targets, retry)
}

type localTextEmbeddingProvider struct {
	descriptor document.EmbeddingDescriptor
	embed      func(context.Context) error
	values     []float32
}

func (p localTextEmbeddingProvider) Descriptor() document.EmbeddingDescriptor { return p.descriptor }
func (p localTextEmbeddingProvider) Embed(ctx context.Context, inputs []document.EmbeddingInput, _ document.EmbeddingAuthorization) (document.EmbeddingResult, error) {
	if p.embed != nil {
		if err := p.embed(ctx); err != nil {
			return document.EmbeddingResult{}, err
		}
	}
	values := p.values
	if values == nil {
		values = []float32{1, 0}
	}
	result := document.EmbeddingResult{}
	for _, input := range inputs {
		result.Vectors = append(result.Vectors, document.EmbeddingVector{Key: input.Key, Values: values})
	}
	return result, nil
}

func TestLocalTextBackfillRecoversAfterProviderOutage(t *testing.T) {
	t.Parallel()
	var ready atomic.Bool
	var calls, clock atomic.Int64
	provider := localTextEmbeddingProvider{descriptor: embeddingWorkerDescriptor(t), embed: func(context.Context) error {
		calls.Add(1)
		if !ready.Load() {
			return errors.New("synthetic server unavailable")
		}
		return nil
	}}
	fixture, setup := newLocalTextBackfillFixture(t, newWorkerProvider(t), provider)
	setup.Service.clock = func() time.Time {
		if calls.Load() == 0 {
			return time.Now().UTC()
		}
		return time.Now().UTC().Add(time.Duration(clock.Add(1)) * 10 * time.Second)
	}
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(mediatest.PDF()))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "outage.pdf", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	done := make(chan error, 1)
	go func() { done <- setup.Backfill(ctx, nil) }()
	defer func() { cancel(); require.ErrorIs(t, <-done, ctx.Err()) }()
	profile := setup.Service.profiles["pdf"]
	require.Eventually(t, func() bool {
		jobs, err := fixture.catalog.EmbeddingJobsForVersionProfile(ctx, node.CurrentVersionID, profile.record.Fingerprint)
		return err == nil && len(jobs) == 1 && jobs[0].State == "failed" && jobs[0].FailureCode == store.EmbeddingFailureProviderUnavailable
	}, 10*time.Second, 10*time.Millisecond)
	failedCalls := calls.Load()
	ready.Store(true)
	require.Never(t, func() bool { return calls.Load() != failedCalls }, 2*time.Second, 20*time.Millisecond, "the per-target delay paces durable requeues")
	require.Eventually(t, func() bool {
		jobs, err := fixture.catalog.EmbeddingJobsForVersionProfile(ctx, node.CurrentVersionID, profile.record.Fingerprint)
		return err == nil && len(jobs) == 1 && jobs[0].State == "completed"
	}, 10*time.Second, 10*time.Millisecond)
}

func TestLocalTextPermanentFailureKeepsBackfillDelay(t *testing.T) {
	t.Parallel()
	var attempts, calls atomic.Int64
	provider := localTextEmbeddingProvider{descriptor: embeddingWorkerDescriptor(t), values: []float32{1}, embed: func(context.Context) error {
		calls.Add(1)
		return nil
	}}
	fixture, setup := newLocalTextBackfillFixture(t, newWorkerProvider(t), provider)
	written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(mediatest.PDF()))
	require.NoError(t, err)
	node, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), "invalid-response.pdf", written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
	require.NoError(t, err)
	failed := make(chan error, 1)
	lane := Backfill[store.DocumentSummary]{Name: "local-text", Page: 50, IdleDelay: time.Second,
		List: setup.targets, Key: func(item store.DocumentSummary) string { return item.ContentVersionID },
		Process: func(ctx context.Context, item store.DocumentSummary) error {
			if item.ContentVersionID == node.CurrentVersionID {
				attempts.Add(1)
			}
			err := setup.process(ctx, item)
			if item.ContentVersionID == node.CurrentVersionID {
				select {
				case failed <- err:
				default:
				}
			}
			return err
		}}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	done := make(chan error, 1)
	go func() { done <- lane.Run(ctx) }()
	defer func() { cancel(); require.ErrorIs(t, <-done, ctx.Err()) }()
	select {
	case err := <-failed:
		require.ErrorContains(t, err, string(store.EmbeddingFailureInvalidResponse))
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	require.Never(t, func() bool { return attempts.Load() > 1 }, 2*time.Second, 20*time.Millisecond, "terminal failures retain the first per-target retry delay")
	require.Equal(t, int64(1), calls.Load())
}

func TestLocalTextPublishesDuringBackfillAndOnIdle(t *testing.T) {
	t.Parallel()
	thirdStarted, release := make(chan struct{}), make(chan struct{})
	calls := 0
	provider := localTextEmbeddingProvider{descriptor: embeddingWorkerDescriptor(t), embed: func(ctx context.Context) error {
		calls++
		if calls == 3 {
			close(thirdStarted)
			select {
			case <-release:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	}}
	renderer := newWorkerProvider(t)
	renderer.textForSource = true
	fixture, setup := newLocalTextBackfillFixture(t, renderer, provider)
	for i := range 8 {
		data := append(mediatest.PDF(), []byte(fmt.Sprintf("\n%% source %d", i))...)
		written, err := fixture.blobs.WriteDetailedContext(t.Context(), bytes.NewReader(data))
		require.NoError(t, err)
		_, err = fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), fmt.Sprintf("batch-%d.pdf", i), written.Hash, written.Size, "application/pdf", processingBlobPhysical(t, written))
		require.NoError(t, err)
	}
	indexer, err := vectorworker.NewIndexWorker(vectorworker.IndexWorkerConfig{Catalog: fixture.catalog,
		ReadVectorSet: func(ctx context.Context, member store.VectorIndexMember) ([]byte, error) {
			return fixture.catalog.ReadVectorIndexVectorSet(ctx, fixture.blobs, member)
		}, Mutate: setup.Service.gate.MutateContext, Owner: "index-worker", BuildLease: 5 * time.Minute, ReaderLease: 5 * time.Minute, IdleDelay: time.Second})
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	backfillDone, indexDone := make(chan error, 1), make(chan error, 1)
	go func() { backfillDone <- setup.Backfill(ctx, nil) }()
	go func() { indexDone <- indexer.Run(ctx) }()
	defer func() {
		cancel()
		require.ErrorIs(t, <-backfillDone, ctx.Err())
		require.ErrorIs(t, <-indexDone, ctx.Err())
	}()
	select {
	case <-thirdStarted:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	space := setup.Service.profiles["pdf"].vectorSpaces["chunk"]
	var during store.VectorIndexGenerationRecord
	require.Eventually(t, func() bool {
		var err error
		during, err = fixture.catalog.ActiveVectorIndexGeneration(ctx, space)
		return err == nil && during.RowCount == 2
	}, 5*time.Second, 10*time.Millisecond, "the recurring publisher runs while backfill is still active")
	close(release)
	require.Eventually(t, func() bool {
		generation, err := fixture.catalog.ActiveVectorIndexGeneration(ctx, space)
		return err == nil && generation.RowCount == 8 && generation.ID != during.ID
	}, 5*time.Second, 10*time.Millisecond, "the recurring publisher includes the final documents")
}

func TestLocalTextTargetPageFiltersCachedCandidates(t *testing.T) {
	t.Parallel()
	fixture, setup := newLocalTextBackfillFixture(t, newWorkerProvider(t))
	version, err := fixture.catalog.ContentVersionByID(t.Context(), fixture.versionID)
	require.NoError(t, err)
	for i := range 120 {
		_, err := fixture.catalog.CreateFile(t.Context(), fixture.catalog.RootID(), fmt.Sprintf("cached-%03d.pdf", i), version.BlobHash, version.Size, "application/pdf")
		require.NoError(t, err)
	}
	items, err := fixture.catalog.MissingProcessingTargetsAfter(t.Context(), setup.Service.profiles["pdf"].record.Fingerprint, "", 1, 1000)
	require.NoError(t, err)
	require.Len(t, items, 121)
	setup.skips = make(map[localTextSkipKey]*localTextSkip)
	for _, item := range items[:120] {
		setup.skips[localTextSkipKey{item.ContentVersionID, item.Name}] = &localTextSkip{seen: true}
	}
	targets, err := setup.targets(t.Context(), "", 50)
	require.NoError(t, err)
	require.Equal(t, items[120:], targets)
	targets, err = setup.targets(t.Context(), targets[0].ContentVersionID, 50)
	require.NoError(t, err)
	require.Empty(t, targets)
}

func BenchmarkLocalTextIdleSweep(b *testing.B) {
	fixture, setup := newLocalTextBackfillFixture(b, newWorkerProvider(b), localTextEmbeddingProvider{descriptor: embeddingWorkerDescriptor(b)})
	version, err := fixture.catalog.ContentVersionByID(b.Context(), fixture.versionID)
	require.NoError(b, err)
	written, err := fixture.blobs.WriteDetailedContext(b.Context(), bytes.NewReader(mediatest.PDF()))
	require.NoError(b, err)
	setup.skips = map[localTextSkipKey]*localTextSkip{{version.ID, "source.pdf"}: {seen: true}}
	for i := range 19999 {
		name := fmt.Sprintf("input-%05d.pdf", i)
		if i < 18999 {
			node, err := fixture.catalog.CreateFile(b.Context(), fixture.catalog.RootID(), name, version.BlobHash, version.Size, "application/pdf")
			require.NoError(b, err)
			setup.skips[localTextSkipKey{node.CurrentVersionID, name}] = &localTextSkip{seen: true}
			continue
		}
		node, err := fixture.catalog.CreateFile(b.Context(), fixture.catalog.RootID(), name, written.Hash, written.Size, "application/pdf", processingBlobPhysical(b, written))
		require.NoError(b, err)
		selector := Selector{Profile: setup.Selected, NodeID: node.ID, ContentVersionID: node.CurrentVersionID}
		plan, err := setup.Service.Plan(b.Context(), selector)
		require.NoError(b, err)
		job, err := setup.Service.Start(b.Context(), StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint,
			runOptions: processingRunOptions{deferIndex: true}})
		require.NoError(b, err)
		require.Len(b, job.EmbeddingJobIDs, 1)
		status, err := fixture.catalog.EmbeddingJobByID(b.Context(), job.EmbeddingJobIDs[0])
		require.NoError(b, err)
		require.Equal(b, "completed", status.State)
	}
	b.ResetTimer()
	for b.Loop() {
		targets, err := setup.targets(b.Context(), "", 50)
		require.NoError(b, err)
		require.Empty(b, targets)
	}
}

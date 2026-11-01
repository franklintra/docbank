package main

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"

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

func TestLocalTextCandidatePreservesProvidersAndSelectsManagedModel(t *testing.T) {
	t.Setenv("DOCBANK_TEST_VOYAGE_KEY", "synthetic-secret")
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
	cfg, err = config.Default().WithLocalText()
	require.NoError(t, err)
	for _, managedBefore := range []bool{false, true} {
		e := cfg.EmbeddingProfiles[config.LocalTextProfile]
		e.Dimensions = 512
		e.Runtime.Managed = managedBefore
		cfg.EmbeddingProfiles[config.LocalTextProfile] = e
		_, managed, rebuild, err = localTextCandidate(cfg, config.LocalTextProfile, "local")
		require.NoError(t, err)
		require.True(t, managed)
		require.True(t, rebuild)
	}
	imageConfig, image, descriptor := voyageOriginalEmbeddingConfig(t)
	mixed, err := imageConfig.WithLocalText()
	require.NoError(t, err)
	p = mixed.ProcessingProfiles[config.LocalTextProfile]
	p.Embeddings = append(p.Embeddings, "voyage")
	mixed.ProcessingProfiles[config.LocalTextProfile] = p
	candidate, managed, rebuild, err = localTextCandidate(mixed, config.LocalTextProfile, "local")
	require.NoError(t, err)
	require.True(t, managed)
	require.False(t, rebuild)
	require.Equal(t, image, candidate.EmbeddingProfiles["voyage"])
	switched, err := configureEmbeddingRuntimeBundle(candidate, unavailableEmbeddingBlobs{}, t.TempDir())
	require.NoError(t, err)
	require.Contains(t, switched.registry.Fingerprints(), descriptor.Fingerprint)
	p.Embeddings = []string{"voyage"}
	mixed.ProcessingProfiles[config.LocalTextProfile] = p
	_, managed, rebuild, err = localTextCandidate(mixed, config.LocalTextProfile, "local")
	require.False(t, managed)
	require.False(t, rebuild)
	require.ErrorIs(t, err, processing.ErrProfileNotConfigured)
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
	for _, scenario := range []string{"missing_profile", "malformed"} {
		t.Run(scenario, func(t *testing.T) {
			root := t.TempDir()
			t.Setenv("DOCBANK_HOME", root)
			t.Setenv("DO_NOT_TRACK", "1")
			data := []byte(`{"version":1,"profile":"deleted-profile","server":"local"}`)
			if scenario == "malformed" {
				data = []byte(`{"version":1`)
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

func TestLocalTextNomicRevisionControlsReuseAndDefault(t *testing.T) {
	t.Parallel()
	for _, revision := range []string{"nomic-embed-text-v1", "nomic-embed-text-v1.5"} {
		t.Run(revision, func(t *testing.T) {
			cfg, err := config.Default().WithLocalText()
			require.NoError(t, err)
			cfg.ProcessingProfiles["archive"] = cfg.ProcessingProfiles[config.LocalTextProfile]
			delete(cfg.ProcessingProfiles, config.LocalTextProfile)
			e := cfg.EmbeddingProfiles[config.LocalTextProfile]
			e.Runtime.ModelRevision, e.Runtime.DeploymentEpoch = revision, revision
			cfg.EmbeddingProfiles[config.LocalTextProfile] = e
			candidate, managed, rebuild, err := localTextCandidate(cfg, "archive", "local")
			require.NoError(t, err)
			require.True(t, managed)
			require.Equal(t, revision != "nomic-embed-text-v1.5", rebuild)
			require.Equal(t, "nomic-embed-text-v1.5", candidate.EmbeddingProfiles[config.LocalTextProfile].Runtime.ModelRevision)
			var identities []string
			for _, current := range []config.Config{cfg, candidate} {
				profile, err := config.ManagedEmbeddingProfile(current.EmbeddingProfiles[config.LocalTextProfile], nil)
				require.NoError(t, err)
				provider, err := openaicompat.New(profile, nil, &http.Client{})
				require.NoError(t, err)
				identity, err := provider.VectorIdentity()
				require.NoError(t, err)
				identities = append(identities, identity)
			}
			require.Equal(t, !rebuild, identities[0] == identities[1])
			if rebuild {
				require.Equal(t, "local-text", defaultLocalTextProfile(cfg))
			} else {
				require.Equal(t, "archive", defaultLocalTextProfile(cfg))
			}
		})
	}
}

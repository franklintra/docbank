package main

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/internal/config"
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
				return embedlocal.Installation{}, nil
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
}

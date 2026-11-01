package store

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
	docsqlite "go.kenn.io/docbank/sqlite"
	"go.kenn.io/docbank/sqlite/modernc"
)

func TestPDFFactSearchMatchesRecordedEvidence(t *testing.T) {
	t.Parallel()
	s := newTestStore(t)
	nodes := make([]Node, 4)
	for i, name := range []string{"unknown.pdf", "plain.pdf", "changed.pdf", "script.pdf"} {
		var err error
		nodes[i], err = s.CreateFile(t.Context(), s.RootID(), name, fmt.Sprintf("%064x", i+1), 12, "application/pdf")
		require.NoError(t, err)
		if i == 0 {
			continue
		}
		fields := []document.SourceMetadataFieldV1{}
		for _, fact := range []struct {
			key   string
			value document.SourceMetadataValueV1
		}{
			{"encrypted", document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(i == 2)}},
			{"last_signature_covers_file", document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(i != 2)}},
			{"has_javascript", document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(i == 3)}},
			{"has_launch_actions", document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(i == 2)}},
			{"signature_field_count", document.SourceMetadataValueV1{Kind: document.SourceMetadataInteger, Integer: new(int64(i - 1))}},
			{"embedded_file_count", document.SourceMetadataValueV1{Kind: document.SourceMetadataInteger, Integer: new(int64(i - 1))}},
			{"permissions", document.SourceMetadataValueV1{Kind: document.SourceMetadataStringList, Strings: []string{"Print", "High-quality print flag"}}},
		} {
			fields = append(fields, document.SourceMetadataFieldV1{Key: "pdf.info." + fact.key, Namespace: "pdf.info", SourceField: fact.key, Value: fact.value})
		}
		canonical, _, err := document.MarshalSourceMetadataV1(document.SourceMetadataV1{ContractVersion: document.SourceMetadataContractV1, Fields: fields})
		require.NoError(t, err)
		_, err = s.PublishSourceMetadata(t.Context(), nodes[i].BlobHash, fakeHash("f1"), canonical)
		require.NoError(t, err)
	}
	for _, tc := range []struct {
		filter string
		want   []int64
	}{
		{"pdf_encrypted:true", []int64{nodes[2].ID}},
		{"pdf_encrypted:false", []int64{nodes[1].ID, nodes[3].ID}},
		{"pdf_last_signature_covers_file:true", []int64{nodes[1].ID, nodes[3].ID}},
		{"pdf_has_launch_actions:false", []int64{nodes[1].ID, nodes[3].ID}},
		{"pdf_signature_field_count_min:1", []int64{nodes[2].ID, nodes[3].ID}},
		{"pdf_signature_field_count_max:1", []int64{nodes[1].ID, nodes[2].ID}},
		{"pdf_embedded_file_count_min:1", []int64{nodes[2].ID, nodes[3].ID}},
		{"pdf_embedded_file_count_max:1", []int64{nodes[1].ID, nodes[2].ID}},
		{"pdf_last_signature_covers_file:false", []int64{nodes[2].ID}},
		{"pdf_has_javascript:true", []int64{nodes[3].ID}},
		{"pdf_has_javascript:false", []int64{nodes[1].ID, nodes[2].ID}},
		{"pdf_has_launch_actions:true", []int64{nodes[2].ID}},
		{"pdf_signature_field_count:0", []int64{nodes[1].ID}},
		{"pdf_signature_field_count:1", []int64{nodes[2].ID}},
		{"pdf_signature_field_count_min:1 AND pdf_signature_field_count_max:1", []int64{nodes[2].ID}},
		{"pdf_embedded_file_count:0", []int64{nodes[1].ID}},
		{"pdf_embedded_file_count:2", []int64{nodes[3].ID}},
		{"pdf_embedded_file_count_min:1 AND pdf_embedded_file_count_max:1", []int64{nodes[2].ID}},
		{`pdf_permissions:"Print"`, []int64{nodes[1].ID, nodes[2].ID, nodes[3].ID}},
		{`pdf_permissions:"print"`, nil},
		{`pdf_permissions:"High-quality"`, nil},
		{"pdf_embedded_file_count_min:1 OR pdf_has_javascript:true", []int64{nodes[2].ID, nodes[3].ID}},
		{"pdf_encrypted:(true OR false) AND NOT pdf_has_javascript:true", []int64{nodes[1].ID, nodes[2].ID}},
	} {
		t.Run(tc.filter, func(t *testing.T) {
			hits, _, err := s.SearchPageWithOptions(t.Context(), "", 50, SearchOptions{PDFFilter: tc.filter})
			require.NoError(t, err)
			var got []int64
			for _, hit := range hits {
				got = append(got, hit.Node.ID)
			}
			assert.ElementsMatch(t, tc.want, got)
			compiled, err := compileQuery(t.Context(), compilerQuery(t, tc.filter), nil)
			require.NoError(t, err)
			assert.ElementsMatch(t, tc.want, compiledFixtureIDs(t, s.db, compiled, ""))
		})
	}

	textAlias, err := s.CreateFile(t.Context(), s.RootID(), "notes.txt", nodes[3].BlobHash, 12, "text/plain")
	require.NoError(t, err)
	otherAlias, err := s.CreateFile(t.Context(), s.RootID(), "other.txt", nodes[1].BlobHash, 12, "text/plain")
	require.NoError(t, err)
	for _, node := range []Node{textAlias, otherAlias} {
		require.NoError(t, s.RecordExtraction(t.Context(), ExtractionResult{BlobHash: node.BlobHash, Extractor: "plain-text", ExtractorVersion: 1, Status: ExtractionOK, Text: "review evidence"}))
	}
	contentHits, _, err := s.SearchPageWithOptions(t.Context(), "evidence", 1, SearchOptions{PDFFilter: "pdf_has_javascript:true"})
	require.NoError(t, err)
	require.Len(t, contentHits, 1)
	assert.Equal(t, textAlias.ID, contentHits[0].Node.ID)
	assert.Equal(t, SearchMatchContent, contentHits[0].Match)

	// An active generation replaces earlier observations for the same bytes.
	canonical, _, err := document.MarshalSourceMetadataV1(document.SourceMetadataV1{ContractVersion: document.SourceMetadataContractV1})
	require.NoError(t, err)
	_, err = s.PublishSourceMetadata(t.Context(), nodes[2].BlobHash, fakeHash("f2"), canonical)
	require.NoError(t, err)
	hits, _, err := s.SearchPageWithOptions(t.Context(), "", 1, SearchOptions{PDFFilter: "pdf_encrypted:true"})
	require.NoError(t, err)
	assert.Empty(t, hits)
	for _, fuzzy := range []bool{false, true} {
		opts := SearchOptions{PDFFilter: "pdf_has_javascript:true", MIMEType: "application/pdf"}
		if fuzzy {
			hits, _, _, err = s.SearchDiscoveryPage(t.Context(), "scrpit", 1, opts)
		} else {
			hits, _, err = s.SearchPageWithOptions(t.Context(), "pdf", 1, opts)
		}
		require.NoError(t, err)
		require.Len(t, hits, 1)
		assert.Equal(t, nodes[3].ID, hits[0].Node.ID)
	}
	_, _, err = s.ReplaceContent(t.Context(), nodes[3].ID, nodes[3].Revision, fakeHash("a9"), 12, "application/pdf")
	require.NoError(t, err)
	hits, _, err = s.SearchPageWithOptions(t.Context(), "", 50, SearchOptions{PDFFilter: "pdf_has_javascript:true", MIMEType: "application/pdf"})
	require.NoError(t, err)
	assert.Empty(t, hits, "historical bytes must not match the current PDF")
}

func TestPDFFilterRejectsInvalidOperands(t *testing.T) {
	t.Parallel()
	for _, text := range []string{"", "name:report", "report", "pdf_encrypted:yes", "pdf_encrypted:TRUE", "pdf_embedded_file_count:>=1", "pdf_embedded_file_count:-1", "pdf_embedded_file_count:9007199254740992", "pdf_has_javascript:true*", "pdf_encrypted:(mime:application/pdf)", "pdf_encrypted:true NEAR/5 pdf_has_javascript:false"} {
		assert.Error(t, ValidatePDFFilter(text), text)
	}
}

func TestPDFFilterNegationExcludesDirectories(t *testing.T) {
	t.Parallel()
	s := newTestStore(t)
	_, err := s.Mkdir(t.Context(), s.RootID(), "review-folder")
	require.NoError(t, err)
	file, err := s.CreateFile(t.Context(), s.RootID(), "review.pdf", fakeHash("a1"), 12, "application/pdf")
	require.NoError(t, err)
	for _, text := range []string{"", "review"} {
		hits, _, err := s.SearchPageWithOptions(t.Context(), text, 50, SearchOptions{PDFFilter: "NOT pdf_has_javascript:true"})
		require.NoError(t, err)
		require.Len(t, hits, 1)
		assert.Equal(t, file.ID, hits[0].Node.ID)
	}
}

func TestSourceMetadataPendingCountsCurrentLiveNodes(t *testing.T) {
	t.Parallel()
	s := newTestStore(t)
	ctx := t.Context()
	file, err := s.CreateFile(ctx, s.RootID(), "review.pdf", fakeHash("a1"), 12, "application/pdf")
	require.NoError(t, err)
	_, err = s.CreateFile(ctx, s.RootID(), "alias.txt", file.BlobHash, 12, "text/plain")
	require.NoError(t, err)
	_, err = s.Mkdir(ctx, s.RootID(), "folder")
	require.NoError(t, err)
	trashed, err := s.CreateFile(ctx, s.RootID(), "trashed.pdf", fakeHash("b2"), 12, "application/pdf")
	require.NoError(t, err)
	_, _, err = s.Trash(ctx, trashed.ID, trashed.Revision)
	require.NoError(t, err)
	count, err := s.SourceMetadataPending(ctx, fakeHash("f1"))
	require.NoError(t, err)
	assert.EqualValues(t, 2, count)
	canonical, _, err := document.MarshalSourceMetadataV1(document.SourceMetadataV1{ContractVersion: document.SourceMetadataContractV1})
	require.NoError(t, err)
	_, err = s.PublishSourceMetadata(ctx, file.BlobHash, fakeHash("f1"), canonical)
	require.NoError(t, err)
	count, err = s.SourceMetadataPending(ctx, fakeHash("f1"))
	require.NoError(t, err)
	assert.Zero(t, count, "unknown PDF facts still have extraction coverage")
	_, _, err = s.ReplaceContent(ctx, file.ID, file.Revision, fakeHash("c3"), 12, "application/pdf")
	require.NoError(t, err)
	count, err = s.SourceMetadataPending(ctx, fakeHash("f1"))
	require.NoError(t, err)
	assert.EqualValues(t, 1, count)
	_, err = s.PublishSourceMetadata(ctx, file.BlobHash, fakeHash("f2"), canonical)
	require.NoError(t, err)
	count, err = s.SourceMetadataPending(ctx, fakeHash("f1"))
	require.NoError(t, err)
	assert.EqualValues(t, 2, count, "a historical generation cannot cover the active head")
}

func TestPDFFactSearchRejectsInvalidGenerations(t *testing.T) {
	t.Parallel()
	for _, adapter := range []struct {
		name   string
		driver docsqlite.Driver
	}{
		{"default", DefaultSQLiteDriver()}, {"pure Go", modernc.Driver{}},
	} {
		for _, damage := range []string{"checksum", "noncanonical", "malformed", "unknown contract", "unknown payload contract"} {
			t.Run(adapter.name+"/"+damage, func(t *testing.T) {
				s, err := Open(filepath.Join(t.TempDir(), "evidence.db"), adapter.driver)
				require.NoError(t, err)
				defer func() { require.NoError(t, s.Close()) }()
				ctx := t.Context()
				node, err := s.CreateFile(ctx, s.RootID(), "evidence.pdf", fakeHash("a1"), 12, "application/pdf")
				require.NoError(t, err)
				canonical, _, err := document.MarshalSourceMetadataV1(document.SourceMetadataV1{ContractVersion: document.SourceMetadataContractV1,
					Fields: []document.SourceMetadataFieldV1{{Key: "pdf.info.has_javascript", Namespace: "pdf.info", SourceField: "has_javascript", Value: document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(true)}}}})
				require.NoError(t, err)
				_, err = s.PublishSourceMetadata(ctx, node.BlobHash, fakeHash("f1"), canonical)
				require.NoError(t, err)
				hits, _, err := s.SearchPageWithOptions(ctx, "", 50, SearchOptions{PDFFilter: "pdf_has_javascript:true"})
				require.NoError(t, err)
				require.Len(t, hits, 1)
				pending, err := s.SourceMetadataPending(ctx, fakeHash("f1"))
				require.NoError(t, err)
				require.Zero(t, pending)
				_, err = s.db.ExecContext(ctx, `DROP TRIGGER source_metadata_generations_immutable_update`)
				require.NoError(t, err)
				switch damage {
				case "checksum":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET checksum=?`, fakeHash("bad"))
				case "noncanonical":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET canonical_json=?`, append([]byte(" "), canonical...))
				case "malformed":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET canonical_json=?`, []byte(`{`))
				case "unknown contract":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET contract_version='source-metadata/v2'`)
				case "unknown payload contract":
					unknown := []byte(strings.ReplaceAll(string(canonical), "source-metadata/v1", "source-metadata/v2"))
					// Keep the checksum valid to isolate contract validation.
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET canonical_json=?,checksum=?`, unknown, fmt.Sprintf("%x", sha256.Sum256(unknown)))
				}
				require.NoError(t, err)
				_, _, err = s.ActiveSourceMetadata(ctx, node.BlobHash)
				require.ErrorIs(t, err, ErrSourceMetadataCorrupt)
				hits, _, err = s.SearchPageWithOptions(ctx, "", 50, SearchOptions{PDFFilter: "pdf_has_javascript:true"})
				require.NoError(t, err)
				assert.Empty(t, hits)
				snapshot, err := s.MaterializeQuerySnapshot(ctx, SnapshotRequest{Query: compilerQuery(t, "pdf_has_javascript:true"), SourceMetadataFingerprint: fakeHash("f1")})
				require.NoError(t, err)
				assert.Zero(t, snapshot.Total)
				require.NotNil(t, snapshot.SourceMetadataPending)
				assert.EqualValues(t, 1, *snapshot.SourceMetadataPending)
				pending, err = s.SourceMetadataPending(ctx, fakeHash("f1"))
				require.NoError(t, err)
				assert.EqualValues(t, 1, pending)
			})
		}
	}
}

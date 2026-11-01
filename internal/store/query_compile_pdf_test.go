package store

import (
	"fmt"
	"path/filepath"
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
		{"pdf_embedded_file_count_min:1", []int64{nodes[2].ID, nodes[3].ID}},
		{"pdf_last_signature_covers_file:false", []int64{nodes[2].ID}},
		{"pdf_has_javascript:true", []int64{nodes[3].ID}},
		{"pdf_has_launch_actions:true", []int64{nodes[2].ID}},
		{"pdf_signature_field_count:0", []int64{nodes[1].ID}},
		{"pdf_signature_field_count_max:1", []int64{nodes[1].ID, nodes[2].ID}},
		{"pdf_embedded_file_count:0", []int64{nodes[1].ID}},
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

	_, _, err := s.ReplaceContent(t.Context(), nodes[3].ID, nodes[3].Revision, fakeHash("a9"), 12, "application/pdf")
	require.NoError(t, err)
	hits, _, err := s.SearchPageWithOptions(t.Context(), "", 50, SearchOptions{PDFFilter: "pdf_has_javascript:true", MIMEType: "application/pdf"})
	require.NoError(t, err)
	assert.Empty(t, hits, "historical bytes must not match the current PDF")
}

func TestPDFFilterRejectsInvalidOperands(t *testing.T) {
	t.Parallel()
	for _, text := range []string{"", "name:report", "report", "pdf_encrypted:yes", "pdf_encrypted:TRUE", "pdf_embedded_file_count:>=1", "pdf_embedded_file_count:-1", "pdf_embedded_file_count:9007199254740992", "pdf_has_javascript:true*", "pdf_encrypted:(mime:application/pdf)", "pdf_encrypted:true NEAR/5 pdf_has_javascript:false"} {
		_, err := compilePDFFilter(text)
		assert.Error(t, err, text)
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

func TestPDFFactSearchRejectsInvalidGenerations(t *testing.T) {
	t.Parallel()
	for _, adapter := range []struct {
		name   string
		driver docsqlite.Driver
	}{
		{"default", DefaultSQLiteDriver()}, {"pure Go", modernc.Driver{}},
	} {
		for _, damage := range []string{"checksum", "unknown contract"} {
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
				_, err = s.db.ExecContext(ctx, `DROP TRIGGER source_metadata_generations_immutable_update`)
				require.NoError(t, err)
				switch damage {
				case "checksum":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET checksum=?`, fakeHash("bad"))
				case "unknown contract":
					_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET contract_version='source-metadata/v2'`)
				}
				require.NoError(t, err)
				_, _, err = s.ActiveSourceMetadata(ctx, node.BlobHash)
				require.ErrorIs(t, err, ErrSourceMetadataCorrupt)
				hits, _, err = s.SearchPageWithOptions(ctx, "", 50, SearchOptions{PDFFilter: "pdf_has_javascript:true"})
				require.NoError(t, err)
				assert.Empty(t, hits)
			})
		}
	}
}

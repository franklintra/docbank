//go:build cgo

package store

import (
	"testing"

	sqlite3 "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/document"
)

func TestPDFNamespaceGuardSkipsNonPDFValidation(t *testing.T) {
	t.Parallel()
	s := newTestStore(t)
	ctx := t.Context()
	for i, field := range []document.SourceMetadataFieldV1{
		{Key: "title", Namespace: "image.exif", SourceField: "Title", Value: document.SourceMetadataValueV1{Kind: document.SourceMetadataString, String: new("Synthetic photo")}},
		{Key: "pdf.info.has_javascript", Namespace: "pdf.info", SourceField: "has_javascript", Value: document.SourceMetadataValueV1{Kind: document.SourceMetadataBoolean, Boolean: new(true)}},
	} {
		node, err := s.CreateFile(ctx, s.RootID(), []string{"photo.tiff", "evidence.pdf"}[i], fakeHash([]string{"a1", "b2"}[i]), 12, "application/octet-stream")
		require.NoError(t, err)
		canonical, _, err := document.MarshalSourceMetadataV1(document.SourceMetadataV1{ContractVersion: document.SourceMetadataContractV1, Fields: []document.SourceMetadataFieldV1{field}})
		require.NoError(t, err)
		_, err = s.PublishSourceMetadata(ctx, node.BlobHash, fakeHash("f1"), canonical)
		require.NoError(t, err)
	}
	_, err := s.db.ExecContext(ctx, `DROP TRIGGER source_metadata_generations_immutable_update`)
	require.NoError(t, err)
	_, err = s.db.ExecContext(ctx, `UPDATE source_metadata_generations SET checksum=? WHERE source_sha256=?`, fakeHash("bad"), fakeHash("b2"))
	require.NoError(t, err)
	conn, err := s.db.Conn(ctx)
	require.NoError(t, err)
	defer func() { require.NoError(t, conn.Close()) }()
	calls := 0
	require.NoError(t, conn.Raw(func(raw any) error {
		// Observe the SQLite boundary while retaining the real evidence decoder.
		driverConn, ok := raw.(*sqlite3.SQLiteConn)
		require.True(t, ok)
		return driverConn.RegisterFunc("docbank_source_metadata_json_v1", func(encoded []byte, contract, checksum string) any {
			calls++
			if _, err := document.DecodeSourceMetadataGenerationV1(encoded, contract, checksum); err != nil {
				return nil
			}
			return string(encoded)
		}, false)
	}))
	compiled, err := compileQuery(ctx, compilerQuery(t, "pdf_has_javascript:true"), nil)
	require.NoError(t, err)
	compiledFixtureIDs(t, conn, compiled, "")
	assert.Equal(t, 1, calls, "only the damaged PDF reaches validation")
}

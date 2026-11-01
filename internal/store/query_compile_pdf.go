package store

import (
	"context"
	"strings"

	"go.kenn.io/docbank/internal/query"
)

func compilePDFPredicate(field, value string, start, end int) (compiledQueryFragment, error) {
	operand, err := query.ParsePDFOperand(field, value)
	if err != nil {
		return compiledQueryFragment{}, compileExpressionError(start, end, err.Error())
	}
	kind := query.PDFFieldKind(field)
	base := strings.TrimSuffix(strings.TrimSuffix(field, "_min"), "_max")
	operator := "="
	if strings.HasSuffix(field, "_min") {
		operator = ">="
	}
	if strings.HasSuffix(field, "_max") {
		operator = "<="
	}
	comparison := `json_extract(fact.value, '$.value.` + kind + `') ` + operator + ` ?`
	if kind == "string_list" {
		comparison = `EXISTS (SELECT 1 FROM json_each(json_extract(fact.value, '$.value.strings')) permission WHERE permission.value=?)`
	}
	return compiledQueryFragment{
		sql: `EXISTS (SELECT 1 FROM source_metadata_heads head
   JOIN source_metadata_generations generation ON generation.generation_id=head.generation_id,
   json_each(generation.canonical_json, '$.fields') fact
   WHERE head.source_sha256=cv.blob_hash
   AND json_extract(fact.value, '$.key')=?
   AND json_extract(fact.value, '$.namespace')='pdf.info'
   AND json_extract(fact.value, '$.value.kind')=? AND ` + comparison + `)`,
		usesPDFFacts: true,
		args:         []any{"pdf.info." + strings.TrimPrefix(base, "pdf_"), kind, operand},
	}, nil
}

// ValidatePDFFilter checks the same bounded PDF expression accepted by Query syntax.
func ValidatePDFFilter(text string) error {
	_, err := compilePDFFilter(text)
	return err
}

func compilePDFFilter(text string) (compiledQueryFragment, error) {
	compiled, err := (queryCompiler{pdfOnly: true}).compile(context.Background(), query.Query{
		V: 1, Text: text, Syntax: "advanced", Mode: "lexical",
		Sort: query.Sort{Field: string(DocumentCatalogSortName), Direction: "asc"},
	}, nil)
	if err != nil {
		return compiledQueryFragment{}, err
	}
	return compiled.bind("", nil)
}

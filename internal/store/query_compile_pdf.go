package store

import (
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
		args: []any{"pdf.info." + strings.TrimPrefix(base, "pdf_"), kind, operand},
	}, nil
}

// ValidatePDFFilter checks the same bounded PDF expression accepted by Query syntax.
func ValidatePDFFilter(text string) error {
	_, err := compilePDFFilter(text)
	return err
}

func compilePDFFilter(text string) (compiledQueryFragment, error) {
	syntax, err := query.ParseExpression(text, "advanced")
	if err != nil {
		return compiledQueryFragment{}, err
	}
	return compilePDFExpression(syntax, "")
}

func compilePDFExpression(syntax *query.Expression, field string) (compiledQueryFragment, error) {
	switch syntax.Kind {
	case query.ExpressionAll, query.ExpressionNear:
	case query.ExpressionField:
		if field == "" && query.PDFFieldKind(syntax.Field) != "" {
			return compilePDFExpression(syntax.Children[0], syntax.Field)
		}
	case query.ExpressionAnd, query.ExpressionOr, query.ExpressionNot:
		parts := make([]compiledQueryFragment, 0, len(syntax.Children))
		for _, child := range syntax.Children {
			part, err := compilePDFExpression(child, field)
			if err != nil {
				return compiledQueryFragment{}, err
			}
			parts = append(parts, part)
		}
		if syntax.Kind == query.ExpressionNot {
			return negateCompiledFragment(parts[0]), nil
		}
		operator := " AND "
		if syntax.Kind == query.ExpressionOr {
			operator = " OR "
		}
		return joinCompiledFragments(parts, operator), nil
	case query.ExpressionTerm, query.ExpressionPhrase:
		if field != "" && !syntax.Prefix {
			return compilePDFPredicate(field, syntax.Value, syntax.Start, syntax.End)
		}
	}
	return compiledQueryFragment{}, compileExpressionError(syntax.Start, syntax.End, "PDF filters require PDF fact fields and exact operands")
}

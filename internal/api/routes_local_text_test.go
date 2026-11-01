package api_test

import (
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	"go.kenn.io/docbank/internal/api"
	"go.kenn.io/docbank/internal/processing"
)

func TestLocalTextSetupRouteReviewsConsentAndImportRace(t *testing.T) {
	t.Parallel()
	var saves, starts atomic.Int64
	var startupFailure, unsupported atomic.Bool
	startupFailure.Store(true)
	ts, catalog := newTestServer(t, func(deps *api.Deps) {
		configure := configureProcessingTestServiceWithEmbeddingProvider(t, newProcessingTestEmbeddingProvider(t), true)
		configure(deps)
		setup := &processing.LocalTextSetup{Service: deps.Processing, DefaultProfile: "private",
			Startup: func(context.Context, bool, string) error {
				starts.Add(1)
				if startupFailure.Load() {
					return errors.New("synthetic startup failure")
				}
				return nil
			},
			Save: func(name, mode string) error {
				require.Equal(t, "private", name)
				require.Equal(t, "local", mode)
				saves.Add(1)
				return nil
			},
		}
		setup.Candidate = func(name, mode string) (*processing.Service, bool, bool, error) {
			require.Equal(t, "private", name)
			require.Equal(t, "local", mode)
			if unsupported.Load() {
				return nil, false, false, fmt.Errorf("%w: choose --profile local-text or configure 768 dimensions", processing.ErrLocalTextConfiguration)
			}
			candidate := *deps
			configure(&candidate)
			return candidate.Processing, false, false, nil
		}
		deps.LocalText = setup
	})
	createFileWithContent(t, ts, catalog, "/first.txt", "first synthetic passage\n")
	response, body := do(t, ts, http.MethodPost, "/api/v1/processing/local/plans", nil, api.LocalTextPlanRequest{Server: "local"})
	require.Equal(t, http.StatusOK, response.StatusCode, body)
	var plan api.LocalTextPlan
	require.NoError(t, json.Unmarshal([]byte(body), &plan))
	require.Equal(t, "this vault and new imports", plan.Scope)
	require.Len(t, plan.Flow, 3)
	require.Equal(t, "query_embedding", plan.Flow[2].Capability)
	collection := importCollection(t, ts.URL, ts.Client(), "manual.txt", "synthetic manual", nil)
	quality, body := get(t, ts, "/api/v1/collections/"+collection.IngestID+"/quality", nil)
	require.Equal(t, http.StatusOK, quality.StatusCode, body)
	var coverage struct {
		Collection struct {
			Coverage struct {
				Configuration string `json:"configuration"`
			} `json:"coverage"`
		} `json:"collection"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &coverage))
	require.Equal(t, "unconfigured", coverage.Collection.Coverage.Configuration, "preview leaves default browser search unchanged")
	request := api.LocalTextApplyRequest{Profile: plan.Profile, Server: plan.Server, PlanFingerprint: plan.Fingerprint}
	response, body = do(t, ts, http.MethodPost, "/api/v1/processing/local/setup", nil, request)
	require.Equal(t, http.StatusPreconditionRequired, response.StatusCode, body)
	require.Zero(t, starts.Load())
	request.Consent = true
	response, body = do(t, ts, http.MethodPost, "/api/v1/processing/local/setup", nil, request)
	require.Equal(t, http.StatusInternalServerError, response.StatusCode, body)
	require.Zero(t, saves.Load())
	startupFailure.Store(false)
	createFileWithContent(t, ts, catalog, "/second.txt", "second synthetic passage\n")
	response, body = do(t, ts, http.MethodPost, "/api/v1/processing/local/setup", nil, request)
	require.Equal(t, http.StatusOK, response.StatusCode, body)
	require.Equal(t, int64(1), saves.Load(), "an imported document doesn't invalidate reviewed ongoing scope")
	request.PlanFingerprint = testHash("stale")
	response, body = do(t, ts, http.MethodPost, "/api/v1/processing/local/setup", nil, request)
	require.Equal(t, http.StatusConflict, response.StatusCode, body)
	require.Equal(t, int64(1), saves.Load())
	unsupported.Store(true)
	response, body = do(t, ts, http.MethodPost, "/api/v1/processing/local/plans", nil, api.LocalTextPlanRequest{Server: "local"})
	require.Equal(t, http.StatusUnprocessableEntity, response.StatusCode, body)
	require.Contains(t, body, "configure 768 dimensions")
}

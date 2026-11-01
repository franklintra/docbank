package processing

import (
	"context"
	"crypto/sha256"
	"encoding/json/v2"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"go.kenn.io/docbank/internal/store"
)

var ErrLocalTextConfiguration = errors.New("unsupported local text configuration")

type LocalTextPlan struct {
	Fingerprint        string    `json:"fingerprint"`
	VaultUID           string    `json:"vault_uid"`
	Profile            string    `json:"profile"`
	Server             string    `json:"server"`
	ProfileFingerprint string    `json:"profile_fingerprint"`
	Models             []string  `json:"models"`
	VectorEffect       string    `json:"vector_effect"`
	Managed            bool      `json:"managed"`
	Flow               []FlowHop `json:"flow"`
	RetainedClasses    []string  `json:"retained_classes"`
	Scope              string    `json:"scope"`
}

// LocalTextSetup builds a private service for the reviewed profile. Activation
// happens on daemon restart, so all consumers keep one effective identity.
type LocalTextSetup struct {
	Service        *Service
	Candidate      func(string, string) (*Service, bool, bool, error)
	DefaultProfile string
	Selected       string
	Startup        func(context.Context, bool, string) error
	Save           func(string, string) error
	mu             sync.Mutex
}

func (setup *LocalTextSetup) review(ctx context.Context, name, mode string) (LocalTextPlan, *Service, error) {
	if name == "" {
		name = setup.DefaultProfile
	}
	service, managed, rebuild, err := setup.Candidate(name, mode)
	if err != nil {
		return LocalTextPlan{}, nil, err
	}
	keep := false
	defer func() {
		if !keep {
			service.Stop()
		}
	}()
	profile, ok := service.profiles[name]
	if !ok {
		return LocalTextPlan{}, nil, ErrProfileNotConfigured
	}
	operation, err := service.planForSource(ctx, Selector{Profile: name}, store.Node{}, store.ContentVersion{}, profile)
	if err != nil {
		return LocalTextPlan{}, nil, err
	}
	plan := LocalTextPlan{VaultUID: service.catalog.VaultID(), Profile: name, Server: mode, ProfileFingerprint: profile.record.Fingerprint,
		Managed: managed, Flow: operation.Flow, RetainedClasses: operation.RetainedClasses, Scope: "this vault and new imports",
		VectorEffect: "reuse matching vectors; embed missing inputs"}
	if rebuild {
		plan.VectorEffect = "switching to Nomic uses a new vector space and rebuilds the index"
	}
	for _, binding := range profile.portable.Embeddings {
		plan.Models = append(plan.Models, binding.Model)
	}
	encoded, err := json.Marshal(plan, json.Deterministic(true))
	if err != nil {
		return LocalTextPlan{}, nil, err
	}
	plan.Fingerprint = fmt.Sprintf("%x", sha256.Sum256(encoded))
	keep = true
	return plan, service, nil
}

func (setup *LocalTextSetup) Plan(ctx context.Context, name, mode string) (LocalTextPlan, error) {
	plan, service, err := setup.review(ctx, name, mode)
	if service != nil {
		service.Stop()
	}
	return plan, err
}

func (setup *LocalTextSetup) Apply(ctx context.Context, name, mode, fingerprint string) (LocalTextPlan, error) {
	setup.mu.Lock()
	defer setup.mu.Unlock()
	plan, service, err := setup.review(ctx, name, mode)
	if err != nil {
		return LocalTextPlan{}, err
	}
	defer service.Stop()
	if fingerprint == "" || plan.Fingerprint != fingerprint {
		return LocalTextPlan{}, ErrPlanChanged
	}
	if err := setup.Startup(ctx, plan.Managed, mode); err != nil {
		return LocalTextPlan{}, err
	}
	if err := service.grantProfileConsent(ctx, service.profiles[plan.Profile], nil); err != nil {
		return LocalTextPlan{}, err
	}
	if err := setup.Service.gate.MutateContext(ctx, func() error { return setup.Save(plan.Profile, plan.Server) }); err != nil {
		return LocalTextPlan{}, fmt.Errorf("%w: save local text selection: %w", ErrConsentSaved, err)
	}

	return plan, nil
}

func (setup *LocalTextSetup) targets(ctx context.Context, after string, limit int) ([]store.DocumentSummary, error) {
	profile, ok := setup.Service.profiles[setup.Selected]
	if !ok {
		return nil, nil
	}
	for _, request := range setup.Service.profileConsentRequests(profile) {
		_, err := setup.Service.catalog.AuthorizeProviderOperation(ctx, request)
		state, err := processingConsentState(err)
		if err != nil {
			return nil, err
		}
		if state != "active" {
			return nil, nil
		}
	}
	return setup.Service.catalog.MissingProcessingTargetsAfter(ctx, profile.record.Fingerprint, after, len(profile.portable.Embeddings), limit)
}

func (setup *LocalTextSetup) process(ctx context.Context, item store.DocumentSummary) error {
	selector := Selector{Profile: setup.Selected, NodeID: item.NodeID, ContentVersionID: item.ContentVersionID}
	plan, err := setup.Service.Plan(ctx, selector)
	if err != nil {
		return err
	}
	if plan.RenditionIneligibleReason != "" {
		return nil
	}
	_, err = setup.Service.Start(ctx, StartRequest{Selector: selector, PlanFingerprint: plan.Fingerprint})
	if errors.Is(err, ErrConsentRequired) || errors.Is(err, store.ErrNotFound) || errors.Is(err, ErrPlanChanged) {
		return nil
	}
	return err
}

func (setup *LocalTextSetup) Backfill(ctx context.Context, logger *slog.Logger) error {
	lane := Backfill[store.DocumentSummary]{Name: "local-text", Page: 50, IdleDelay: time.Second, Logger: logger,
		List: setup.targets, Key: func(item store.DocumentSummary) string { return item.ContentVersionID }, Process: setup.process}
	return lane.Run(ctx)
}

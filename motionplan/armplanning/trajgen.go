package armplanning

import (
	"context"
	"fmt"
	"time"

	"github.com/pkg/errors"
	"go.viam.com/utils/trace"
	"gorgonia.org/tensor"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
)

// TrajGenConfig holds configuration for the trajectory generator ML model service.
type TrajGenConfig struct {
	Service                            string   `json:"service"`
	PathToleranceDeltaRads             *float64 `json:"path_tolerance_delta_rads,omitempty"`
	PathColinearizationRatio           *float64 `json:"path_colinearization_ratio,omitempty"`
	WaypointDeduplicationToleranceRads *float64 `json:"waypoint_deduplication_tolerance_rads,omitempty"`
	VelocityLimitsRadsPerSec           float64  `json:"velocity_limits_rads_per_sec,omitempty"`
	AccelerationLimitsRadsPerSec2      float64  `json:"acceleration_limits_rads_per_sec2,omitempty"`
	SamplingFreqHz                     *float64 `json:"trajectory_sampling_freq_hz,omitempty"`
}

func (cfg *TrajGenConfig) Validate(path string) ([]string, error) {
	if cfg.VelocityLimitsRadsPerSec <= 0 {
		return nil, fmt.Errorf("need positive velocity_limits_rads_per_sec if using trajectory_generator, got %v", cfg.VelocityLimitsRadsPerSec)
	}
	if cfg.AccelerationLimitsRadsPerSec2 <= 0 {
		return nil, fmt.Errorf("need positive acceleration_limits_rads_per_sec2 if using trajectory_generator, got %v", cfg.AccelerationLimitsRadsPerSec2)
	}
	if cfg.Service == "" {
		return nil, resource.NewConfigValidationFieldRequiredError(path, "service")
	}
	return []string{cfg.Service}, nil
}

const (
	defaultTrajGenPathToleranceDeltaRads             = 0.1
	defaultTrajGenWaypointDeduplicationToleranceRads = 1e-3
	defaultTrajGenSamplingFreqHz                     = 10.0
	defaultTrajGenPathColinearizationRatio           = 0.0
)

// TrajGen holds a resolved trajectory generator ML model service along with its configuration.
type TrajGen struct {
	trajGen                            mlmodel.Service
	PathToleranceDeltaRads             float64 `json:"path_tolerance_delta_rads"`
	PathColinearizationRatio           float64 `json:"path_colinearization_ratio"`
	WaypointDeduplicationToleranceRads float64 `json:"waypoint_deduplication_tolerance_rads"`
	VelocityLimitsRadsPerSec           float64 `json:"velocity_limits_rads_per_sec"`
	AccelerationLimitsRadsPerSec2      float64 `json:"acceleration_limits_rads_per_sec2"`
	SamplingFreqHz                     float64 `json:"trajectory_sampling_freq_hz"`
}

func applyDefault(v *float64, def float64) float64 {
	if v == nil {
		return def
	}
	return *v
}

// NewTrajGen constructs a TrajGen from an mlmodel service and configuration fields,
// applying defaults for any nil optional values.
func NewTrajGen(
	svc mlmodel.Service,
	pathToleranceDeltaRads *float64,
	pathColinearizationRatio *float64,
	waypointDeduplicationToleranceRads *float64,
	velocityLimitsRadsPerSec float64,
	accelerationLimitsRadsPerSec2 float64,
	samplingFreqHz *float64,
) *TrajGen {
	return &TrajGen{
		trajGen:                            svc,
		PathToleranceDeltaRads:             applyDefault(pathToleranceDeltaRads, defaultTrajGenPathToleranceDeltaRads),
		PathColinearizationRatio:           applyDefault(pathColinearizationRatio, defaultTrajGenPathColinearizationRatio),
		WaypointDeduplicationToleranceRads: applyDefault(waypointDeduplicationToleranceRads, defaultTrajGenWaypointDeduplicationToleranceRads),
		VelocityLimitsRadsPerSec:           velocityLimitsRadsPerSec,
		AccelerationLimitsRadsPerSec2:      accelerationLimitsRadsPerSec2,
		SamplingFreqHz:                     applyDefault(samplingFreqHz, defaultTrajGenSamplingFreqHz),
	}
}

// inferTrajGen sends the waypoints to the trajectory generator service and returns the resulting
// densely-sampled trajectory. Returns nil if the service indicates the arm is already at the goal.
func inferTrajGen(
	ctx context.Context,
	fs *referenceframe.FrameSystem,
	trajAsInps []*referenceframe.LinearInputs,
	tg *TrajGen,
) ([]*referenceframe.LinearInputs, error) {
	if len(trajAsInps) == 0 {
		return trajAsInps, nil
	}

	schema, err := trajAsInps[0].GetSchema(fs)
	if err != nil {
		return nil, err
	}

	dof := len(trajAsInps[0].GetLinearizedInputs())
	nWaypoints := len(trajAsInps)

	waypoints := make([]float64, 0, nWaypoints*dof)
	for _, li := range trajAsInps {
		waypoints = append(waypoints, li.GetLinearizedInputs()...)
	}

	velLimits := make([]float64, dof)
	accelLimits := make([]float64, dof)
	for i := range dof {
		velLimits[i] = tg.VelocityLimitsRadsPerSec
		accelLimits[i] = tg.AccelerationLimitsRadsPerSec2
	}

	outMap, err := tg.trajGen.Infer(ctx, ml.Tensors{
		"waypoints_rads": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(nWaypoints, dof),
			tensor.WithBacking(waypoints),
		),
		"velocity_limits_rads_per_sec": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(dof),
			tensor.WithBacking(velLimits),
		),
		"acceleration_limits_rads_per_sec2": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(dof),
			tensor.WithBacking(accelLimits),
		),
		"path_tolerance_delta_rads": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(1),
			tensor.WithBacking([]float64{tg.PathToleranceDeltaRads}),
		),
		"path_colinearization_ratio": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(1),
			tensor.WithBacking([]float64{tg.PathColinearizationRatio}),
		),
		"waypoint_deduplication_tolerance_rads": tensor.New(
			tensor.Of(tensor.Float64),
			tensor.WithShape(1),
			tensor.WithBacking([]float64{tg.WaypointDeduplicationToleranceRads}),
		),
		"trajectory_sampling_freq_hz": tensor.New(
			tensor.Of(tensor.Int64),
			tensor.WithShape(1),
			tensor.WithBacking([]int64{int64(tg.SamplingFreqHz)}),
		),
	})
	if err != nil {
		return nil, err
	}

	configsTensor, ok := outMap["configurations_rads"]
	if !ok {
		// Service returns an empty map when fewer than 2 distinct waypoints remain after
		// deduplication -- the arm is already at the goal.
		return nil, nil
	}

	configsData := configsTensor.Data().([]float64)
	nSamples := configsTensor.Shape()[0]
	result := make([]*referenceframe.LinearInputs, nSamples)
	for i := range nSamples {
		li, err := schema.FloatsToInputs(configsData[i*dof : (i+1)*dof])
		if err != nil {
			return nil, err
		}
		result[i] = li
	}
	return result, nil
}

// PlanMotionTrajGen plans a motion from a provided plan request using a trajectory generator.
func PlanMotionTrajGen(ctx context.Context, parentLogger logging.Logger, request *PlanRequest, trajGen *TrajGen) (motionplan.Plan, *PlanMeta, error) {
	logger := parentLogger.Sublogger("mp")

	start := time.Now()
	meta := &PlanMeta{}
	ctx, span := trace.StartSpan(ctx, "PlanMotion")
	defer func() {
		meta.Duration = time.Since(start)
		span.End()
	}()

	if err := request.validatePlanRequest(); err != nil {
		return nil, meta, err
	}
	logger.CDebugf(ctx, "constraint specs for this step: %v", request.Constraints)
	logger.CDebugf(ctx, "motion config for this step: %v", request.PlannerOptions)
	logger.CDebugf(ctx, "start position: %v", request.StartState.structuredConfiguration)

	if request.PlannerOptions == nil {
		request.PlannerOptions = NewBasicPlannerOptions()
	}

	if request.StartState.structuredConfiguration == nil {
		return nil, meta, errors.New("must populate start state configuration")
	}

	sfPlanner, err := newPlanManager(ctx, logger, request, meta)
	if err != nil {
		return nil, meta, err
	}

	trajAsInps, goalsProcessed, err := sfPlanner.planMultiWaypoint(ctx)
	if err != nil {
		if request.PlannerOptions.ReturnPartialPlan {
			meta.Partial = true
			meta.PartialError = err
			logger.Infof("returning partial plan, error: %v", err)
		} else {
			return nil, meta, err
		}
	}

	meta.GoalsProcessed = goalsProcessed

	trajAsInps, err = inferTrajGen(ctx, request.FrameSystem, trajAsInps, trajGen)
	if err != nil {
		return nil, meta, err
	}
	if trajAsInps == nil {
		trajAsInps = []*referenceframe.LinearInputs{}
	}

	t, err := motionplan.NewSimplePlanFromTrajectory(trajAsInps, request.FrameSystem)
	if err != nil {
		return nil, meta, err
	}

	return t, meta, nil
}

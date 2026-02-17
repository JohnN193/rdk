// Package builtin implements a remote control for a base.
package builtin

import (
	"context"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/geo/r3"
	"github.com/pkg/errors"
	vutils "go.viam.com/utils"

	"go.viam.com/rdk/components/base"
	"go.viam.com/rdk/components/input"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/baseremotecontrol"
	"go.viam.com/rdk/session"
)

// Constants for the system including the max speed and angle (TBD: allow to be set as config vars)
// as well as the various control modes including oneJoystick (control via a joystick), triggerSpeed
// (triggers control speed and joystick angle), button (four buttons X, Y, A, B to  control speed and
// angle) and arrow (arrows buttons used to control speed and angle).
const (
	joyStickControl = controlMode(iota)
	triggerSpeedControl
	buttonControl
	arrowControl
	droneControl
	funBaseControl
)

var modes = []string{"joystickControl", "triggerSpeedControl", "buttonControl", "arrowControl", "droneControl", "funBaseControl"}

func init() {
	resource.RegisterService(baseremotecontrol.API, resource.DefaultServiceModel, resource.Registration[baseremotecontrol.Service, *Config]{
		Constructor: NewBuiltIn,
	})
}

// ControlMode is the control type for the remote control.
type controlMode uint8

// Config describes how to configure the service.
type Config struct {
	BaseName            string                `json:"base"`
	InputControllerName string                `json:"input_controller"`
	ControlModeName     string                `json:"control_mode,omitempty"`
	MaxAngularVelocity  float64               `json:"max_angular_deg_per_sec,omitempty"`
	MaxLinearVelocity   float64               `json:"max_linear_mm_per_sec,omitempty"`
	FunCommands         map[string]FunCommand `json:"fun_commands,omitempty"`
}

type FunCommand struct {
	Command        string      `json:"cmd,omitempty"`
	DoCommandInput interface{} `json:"input,omitempty"`
	EventType      string      `json:"event_type,omitempty"`
}

// Validate creates the list of implicit dependencies.
func (conf *Config) Validate(path string) ([]string, []string, error) {
	var deps []string
	if conf.InputControllerName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "input_controller")
	}
	deps = append(deps, conf.InputControllerName)

	if conf.BaseName == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "base")
	}
	deps = append(deps, conf.BaseName)

	if conf.ControlModeName != "" {
		configModeExists := false
		for _, mode := range modes {
			if mode == conf.ControlModeName {
				configModeExists = true
				break
			}
		}

		if !configModeExists {
			return nil, nil, resource.NewConfigValidationError(path, errors.Errorf("Control mode '%s' is not in %v", conf.ControlModeName, modes))
		}
	}

	if conf.ControlModeName == "funBaseControl" {
		validControls := map[input.Control]bool{
			input.AbsoluteX: false, input.AbsoluteY: false, input.AbsoluteZ: true,
			input.AbsoluteRX: false, input.AbsoluteRY: false, input.AbsoluteRZ: true,
			input.AbsoluteHat0X: true, input.AbsoluteHat0Y: true,
			input.ButtonSouth: true, input.ButtonEast: true, input.ButtonWest: true, input.ButtonNorth: true,
			input.ButtonLT: true, input.ButtonRT: true, input.ButtonLT2: true, input.ButtonRT2: true,
			input.ButtonLThumb: true, input.ButtonRThumb: true,
			input.ButtonSelect: true, input.ButtonStart: true, input.ButtonMenu: true,
			input.ButtonRecord: true, input.ButtonEStop: true,
			input.AbsolutePedalAccelerator: true, input.AbsolutePedalBrake: true, input.AbsolutePedalClutch: true,
		}
		validEventTypes := map[input.EventType]bool{
			input.ButtonPress: true, input.ButtonRelease: true,
			input.ButtonHold: true, input.ButtonChange: true,
			input.PositionChangeAbs: true, input.PositionChangeRel: true,
		}
		for k, fc := range conf.FunCommands {
			if !validControls[input.Control(k)] {
				return nil, nil, resource.NewConfigValidationError(path,
					errors.Errorf("fun_commands key '%s' is not a valid input control", k))
			}
			if fc.EventType != "" && !validEventTypes[input.EventType(fc.EventType)] {
				return nil, nil, resource.NewConfigValidationError(path,
					errors.Errorf("fun_commands key '%s' has invalid event_type '%s'", k, fc.EventType))
			}
		}
	}

	return deps, nil, nil
}

// builtIn is the structure of the remote service.
type builtIn struct {
	resource.Named

	mu              sync.RWMutex
	base            base.Base
	inputController input.Controller
	controlMode     controlMode
	config          *Config

	state                   throttleState
	logger                  logging.Logger
	cancel                  func()
	cancelCtx               context.Context
	activeBackgroundWorkers sync.WaitGroup
	events                  chan (struct{})
	funCmdQueue             chan map[string]interface{}
	instance                atomic.Int64
}

// NewBuiltIn returns a new remote control service for the given robot.
func NewBuiltIn(
	ctx context.Context,
	deps resource.Dependencies,
	conf resource.Config,
	logger logging.Logger,
) (baseremotecontrol.Service, error) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	remoteSvc := &builtIn{
		Named:       conf.ResourceName().AsNamed(),
		logger:      logger,
		cancelCtx:   cancelCtx,
		cancel:      cancel,
		events:      make(chan struct{}, 1),
		funCmdQueue: make(chan map[string]interface{}, 1),
	}
	remoteSvc.state.init()
	if err := remoteSvc.Reconfigure(ctx, deps, conf); err != nil {
		return nil, err
	}
	remoteSvc.eventProcessor()

	return remoteSvc, nil
}

func (svc *builtIn) Reconfigure(
	ctx context.Context,
	deps resource.Dependencies,
	conf resource.Config,
) error {
	svcConfig, err := resource.NativeConfig[*Config](conf)
	if err != nil {
		return err
	}
	base1, err := base.FromProvider(deps, svcConfig.BaseName)
	if err != nil {
		return err
	}
	controller, err := input.FromProvider(deps, svcConfig.InputControllerName)
	if err != nil {
		return err
	}

	var controlMode1 controlMode
	switch svcConfig.ControlModeName {
	case "triggerSpeedControl":
		controlMode1 = triggerSpeedControl
	case "buttonControl":
		controlMode1 = buttonControl
	case "joystickControl":
		controlMode1 = joyStickControl
	case "droneControl":
		controlMode1 = droneControl
	case "funBaseControl":
		controlMode1 = funBaseControl
	default:
		controlMode1 = arrowControl
	}

	svc.mu.Lock()
	svc.base = base1
	svc.inputController = controller
	svc.controlMode = controlMode1
	svc.config = svcConfig
	svc.mu.Unlock()
	svc.instance.Add(1)

	if err := svc.registerCallbacks(ctx, &svc.state); err != nil {
		return errors.Errorf("error with starting remote control service: %q", err)
	}

	return nil
}

// registerCallbacks registers events from controller to base.
func (svc *builtIn) registerCallbacks(ctx context.Context, state *throttleState) error {
	var lastTS time.Time
	lastTSPerEvent := map[input.Control]map[input.EventType]time.Time{}
	var onlyOneAtATime sync.Mutex

	instance := svc.instance.Load()

	updateLastEvent := func(event input.Event) bool {
		if event.Time.After(lastTS) {
			lastTS = event.Time
		}
		if event.Time.Before(lastTSPerEvent[event.Control][event.Event]) {
			return false
		}
		lastTSPerEventControl := lastTSPerEvent[event.Control]
		if lastTSPerEventControl == nil {
			lastTSPerEventControl = map[input.EventType]time.Time{}
			lastTSPerEvent[event.Control] = lastTSPerEventControl
		}
		lastTSPerEventControl[event.Event] = event.Time
		return true
	}

	remoteCtl := func(ctx context.Context, event input.Event) {
		onlyOneAtATime.Lock()
		defer onlyOneAtATime.Unlock()

		if svc.instance.Load() != instance {
			return
		}

		if svc.cancelCtx.Err() != nil {
			return
		}

		if !updateLastEvent(event) {
			return
		}

		svc.processEvent(ctx, state, event)
	}

	connect := func(ctx context.Context, event input.Event) {
		onlyOneAtATime.Lock()
		defer onlyOneAtATime.Unlock()

		if svc.instance.Load() != instance {
			return
		}

		// Connect and Disconnect events should both stop the base completely.
		svc.mu.RLock()
		defer svc.mu.RUnlock()
		err := svc.base.Stop(ctx, map[string]interface{}{})
		if err != nil {
			svc.logger.CError(ctx, err)
		}

		if !updateLastEvent(event) {
			return
		}
	}

	for _, control := range svc.ControllerInputs() {
		if err := func() error {
			svc.mu.RLock()
			defer svc.mu.RUnlock()
			var err error
			eventTypes := []input.EventType{input.PositionChangeAbs}
			if svc.controlMode == buttonControl || strings.HasPrefix(string(control), "Button") {
				eventTypes = []input.EventType{input.ButtonChange}
			}
			err = svc.inputController.RegisterControlCallback(
				ctx,
				control,
				eventTypes,
				remoteCtl,
				map[string]interface{}{},
			)
			if err != nil {
				return err
			}
			err = svc.inputController.RegisterControlCallback(ctx,
				control,
				[]input.EventType{input.Connect, input.Disconnect},
				connect,
				map[string]interface{}{},
			)
			if err != nil {
				return err
			}
			return nil
		}(); err != nil {
			return err
		}
	}
	return nil
}

// Close out of all remote control related systems.
func (svc *builtIn) Close(_ context.Context) error {
	svc.cancel()
	svc.activeBackgroundWorkers.Wait()
	return nil
}

// ControllerInputs returns the list of inputs from the controller that are being monitored for that control mode.
func (svc *builtIn) ControllerInputs() []input.Control {
	svc.mu.RLock()
	defer svc.mu.RUnlock()
	switch svc.controlMode {
	case triggerSpeedControl:
		return []input.Control{input.AbsoluteX, input.AbsoluteZ, input.AbsoluteRZ}
	case arrowControl:
		return []input.Control{input.AbsoluteHat0X, input.AbsoluteHat0Y}
	case buttonControl:
		return []input.Control{input.ButtonNorth, input.ButtonSouth, input.ButtonEast, input.ButtonWest}
	case joyStickControl:
		return []input.Control{input.AbsoluteX, input.AbsoluteY}
	case droneControl:
		return []input.Control{input.AbsoluteX, input.AbsoluteY, input.AbsoluteRX, input.AbsoluteRY}
	case funBaseControl:
		controls := []input.Control{input.AbsoluteX, input.AbsoluteY, input.AbsoluteRX, input.AbsoluteRY}
		for k := range svc.config.FunCommands {
			controls = append(controls, input.Control(k))
		}
		return controls
	}
	return []input.Control{}
}

func (svc *builtIn) eventProcessor() {
	var currentLinear, currentAngular r3.Vector
	var nextLinear, nextAngular r3.Vector
	var inRetry bool

	svc.activeBackgroundWorkers.Add(1)
	vutils.ManagedGo(func() {
		for {
			if svc.cancelCtx.Err() != nil {
				return
			}

			if inRetry {
				select {
				case <-svc.cancelCtx.Done():
				case <-svc.events:
				default:
				}
			} else {
				select {
				case <-svc.cancelCtx.Done():
				case <-svc.events:
				}
			}
			svc.state.mu.Lock()
			nextLinear, nextAngular = svc.state.linearThrottle, svc.state.angularThrottle
			svc.state.mu.Unlock()

			if func() bool {
				svc.mu.RLock()
				defer svc.mu.RUnlock()

				if currentLinear != nextLinear || currentAngular != nextAngular {
					if svc.config.MaxAngularVelocity > 0 && svc.config.MaxLinearVelocity > 0 {
						if err := svc.base.SetVelocity(
							svc.cancelCtx,
							r3.Vector{
								X: svc.config.MaxLinearVelocity * nextLinear.X,
								Y: svc.config.MaxLinearVelocity * nextLinear.Y,
								Z: svc.config.MaxLinearVelocity * nextLinear.Z,
							},
							r3.Vector{
								X: svc.config.MaxAngularVelocity * nextAngular.X,
								Y: svc.config.MaxAngularVelocity * nextAngular.Y,
								Z: svc.config.MaxAngularVelocity * nextAngular.Z,
							},
							nil,
						); err != nil {
							svc.logger.Errorw("error setting velocity", "error", err)
							if !vutils.SelectContextOrWait(svc.cancelCtx, 10*time.Millisecond) {
								return true
							}
							inRetry = true
							return false
						}
					} else {
						if err := svc.base.SetPower(svc.cancelCtx, nextLinear, nextAngular, nil); err != nil {
							svc.logger.Errorw("error setting power", "error", err)
							if !vutils.SelectContextOrWait(svc.cancelCtx, 10*time.Millisecond) {
								return true
							}
							inRetry = true
							return false
						}
					}
					inRetry = false

					currentLinear = nextLinear
					currentAngular = nextAngular
				}

				return false
			}() {
				return
			}

			select {
			case cmd := <-svc.funCmdQueue:
				svc.logger.Infow("executing fun command from queue", "cmd", cmd)
				svc.mu.RLock()
				if _, err := svc.base.DoCommand(svc.cancelCtx, cmd); err != nil {
					svc.logger.Errorw("error executing fun command", "error", err)
				}
				svc.mu.RUnlock()
			default:
			}
		}
	}, svc.activeBackgroundWorkers.Done)
}

func (svc *builtIn) processEvent(ctx context.Context, state *throttleState, event input.Event) {
	// Order of who processes what event is *not* guaranteed. It depends on the mutex
	// fairness mode. Ordering logic must be handled at a higher level in the robot.
	// Other than that, values overwrite each other.
	state.mu.Lock()
	oldLinear := state.linearThrottle
	oldAngular := state.angularThrottle
	newLinear := oldLinear
	newAngular := oldAngular

	svc.mu.RLock()
	defer svc.mu.RUnlock()

	switch svc.controlMode {
	case joyStickControl:
		newLinear.Y, newAngular.Z = oneJoyStickEvent(event, state.linearThrottle.Y, state.angularThrottle.Z)
	case droneControl:
		newLinear, newAngular = droneEvent(event, state.linearThrottle, state.angularThrottle)
	case funBaseControl:
		switch event.Control {
		case input.AbsoluteX, input.AbsoluteY, input.AbsoluteRX, input.AbsoluteRY:
			newLinear, newAngular = funBaseEvent(event, state.linearThrottle, state.angularThrottle)
		case input.AbsoluteHat0X, input.AbsoluteHat0Y, input.AbsoluteRZ, input.AbsoluteZ, input.ButtonEStop,
			input.ButtonEast, input.ButtonLT, input.ButtonLT2, input.ButtonLThumb, input.ButtonMenu, input.ButtonNorth,
			input.ButtonRT, input.ButtonRT2, input.ButtonRThumb, input.ButtonRecord, input.ButtonSelect,
			input.ButtonSouth, input.ButtonStart, input.ButtonWest, input.AbsolutePedalAccelerator,
			input.AbsolutePedalBrake, input.AbsolutePedalClutch:
			svc.logger.Infow("fun control event", "control", event.Control, "value", event.Value)
			if funCmd, ok := svc.config.FunCommands[string(event.Control)]; ok {
				expectedEventType := input.EventType(funCmd.EventType)
				if expectedEventType == "" {
					expectedEventType = input.ButtonPress
				}
				if event.Event != expectedEventType {
					svc.logger.Debugw("fun command event type mismatch", "control", event.Control, "expected", expectedEventType, "got", event.Event)
				} else {
					svc.logger.Infow("enqueueing fun command", "cmd", funCmd.Command, "input", funCmd.DoCommandInput)
					select {
					case svc.funCmdQueue <- map[string]interface{}{funCmd.Command: funCmd.DoCommandInput}:
						svc.logger.Infow("fun command enqueued")
					default:
						svc.logger.Warnw("fun command queue full, dropping command")
					}
				}
			} else {
				svc.logger.Debugw("no fun command configured for control", "control", event.Control)
			}
			fallthrough
		default:
			newLinear = oldLinear
			newAngular = oldAngular
		}

	case triggerSpeedControl:
		newLinear.Y, newAngular.Z = triggerSpeedEvent(event, state.linearThrottle.Y, state.angularThrottle.Z)
	case buttonControl:
		newLinear.Y, newAngular.Z, state.buttons = buttonControlEvent(event, state.buttons)
	case arrowControl:
		newLinear.Y, newAngular.Z, state.arrows = arrowEvent(event, state.arrows)
	}
	state.linearThrottle = newLinear
	state.angularThrottle = newAngular
	state.mu.Unlock()

	if similar(newLinear, oldLinear, .05) && similar(newAngular, oldAngular, .05) && len(svc.funCmdQueue) == 0 {
		svc.logger.Debugw("skipping event signal, no changes", "control", event.Control)
		return
	}
	svc.logger.Debugw("signaling event processor", "control", event.Control, "funCmdQueueLen", len(svc.funCmdQueue))

	// If we do not manage to send the event, that means the processor
	// is working and it is about to see our state change anyway. This
	// actls like a condition variable signal.
	select {
	case <-ctx.Done():
	case svc.events <- struct{}{}:
	default:
	}

	session.SafetyMonitor(ctx, svc.base)
}

// triggerSpeedEvent takes inputs from the gamepad allowing the triggers to control speed and the left joystick to
// control the angle.
func triggerSpeedEvent(event input.Event, speed, angle float64) (float64, float64) {
	switch event.Control {
	case input.AbsoluteZ:
		speed -= 0.05
		speed = math.Max(-1, speed)
	case input.AbsoluteRZ:
		speed += 0.05
		speed = math.Min(1, speed)
	case input.AbsoluteX:
		angle = event.Value
	case input.AbsoluteHat0X, input.AbsoluteHat0Y, input.AbsoluteRX, input.AbsoluteRY, input.AbsoluteY,
		input.ButtonEStop, input.ButtonEast, input.ButtonLT, input.ButtonLT2, input.ButtonLThumb, input.ButtonMenu,
		input.ButtonNorth, input.ButtonRT, input.ButtonRT2, input.ButtonRThumb, input.ButtonRecord,
		input.ButtonSelect, input.ButtonSouth, input.ButtonStart, input.ButtonWest, input.AbsolutePedalAccelerator,
		input.AbsolutePedalBrake, input.AbsolutePedalClutch:
		fallthrough
	default:
	}

	return speed, angle
}

// buttonControlEvent takes inputs from the gamepad allowing the X and B buttons to control speed and Y and A buttons to control angle.
func buttonControlEvent(event input.Event, buttons map[input.Control]bool) (float64, float64, map[input.Control]bool) {
	var speed float64
	var angle float64

	switch event.Event {
	case input.ButtonPress:
		buttons[event.Control] = true
	case input.ButtonRelease:
		buttons[event.Control] = false
	case input.AllEvents, input.ButtonChange, input.ButtonHold, input.Connect, input.Disconnect,
		input.PositionChangeAbs, input.PositionChangeRel:
		fallthrough
	default:
	}

	if buttons[input.ButtonNorth] == buttons[input.ButtonSouth] {
		speed = 0.0
	} else {
		if buttons[input.ButtonNorth] {
			speed = 1.0
		} else {
			speed = -1.0
		}
	}

	if buttons[input.ButtonEast] == buttons[input.ButtonWest] {
		angle = 0.0
	} else {
		if buttons[input.ButtonEast] {
			angle = -1.0
		} else {
			angle = 1.0
		}
	}

	return speed, angle, buttons
}

// arrowControlEvent takes inputs from the gamepad allowing the arrow buttons to control speed and angle.
func arrowEvent(event input.Event, arrows map[input.Control]float64) (float64, float64, map[input.Control]float64) {
	arrows[event.Control] = -1.0 * event.Value

	speed := arrows[input.AbsoluteHat0Y]
	angle := arrows[input.AbsoluteHat0X]

	return speed, angle, arrows
}

// oneJoyStickEvent (default) takes inputs from the gamepad allowing the left joystick to control speed and angle.
func oneJoyStickEvent(event input.Event, y, x float64) (float64, float64) {
	switch event.Control {
	case input.AbsoluteY:
		y = -1.0 * event.Value
	case input.AbsoluteX:
		x = -1.0 * event.Value
	case input.AbsoluteHat0X, input.AbsoluteHat0Y, input.AbsoluteRX, input.AbsoluteRY, input.AbsoluteRZ,
		input.AbsoluteZ, input.ButtonEStop, input.ButtonEast, input.ButtonLT, input.ButtonLT2, input.ButtonLThumb,
		input.ButtonMenu, input.ButtonNorth, input.ButtonRT, input.ButtonRT2, input.ButtonRThumb,
		input.ButtonRecord, input.ButtonSelect, input.ButtonSouth, input.ButtonStart, input.ButtonWest, input.AbsolutePedalAccelerator,
		input.AbsolutePedalBrake, input.AbsolutePedalClutch:
		fallthrough
	default:
	}

	return scaleThrottle(y), scaleThrottle(x)
}

// right joystick is forward/back, strafe right/left
// left joystick is spin right/left & up/down.
func droneEvent(event input.Event, linear, angular r3.Vector) (r3.Vector, r3.Vector) {
	switch event.Control {
	case input.AbsoluteX:
		angular.Z = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteY:
		linear.Z = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteRX:
		linear.X = scaleThrottle(event.Value)
	case input.AbsoluteRY:
		linear.Y = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteHat0X, input.AbsoluteHat0Y, input.AbsoluteRZ, input.AbsoluteZ, input.ButtonEStop,
		input.ButtonEast, input.ButtonLT, input.ButtonLT2, input.ButtonLThumb, input.ButtonMenu, input.ButtonNorth,
		input.ButtonRT, input.ButtonRT2, input.ButtonRThumb, input.ButtonRecord, input.ButtonSelect,
		input.ButtonSouth, input.ButtonStart, input.ButtonWest, input.AbsolutePedalAccelerator,
		input.AbsolutePedalBrake, input.AbsolutePedalClutch:
		fallthrough
	default:
	}

	return linear, angular
}

// right joystick is forward/back, strafe right/left
// left joystick is spin right/left & up/down.
func funBaseEvent(event input.Event, linear, angular r3.Vector) (r3.Vector, r3.Vector) {
	switch event.Control {
	case input.AbsoluteX:
		linear.X = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteY:
		linear.Y = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteRX:
		angular.Z = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteRY:
		angular.X = scaleThrottle(-1.0 * event.Value)
	case input.AbsoluteHat0X, input.AbsoluteHat0Y, input.AbsoluteRZ, input.AbsoluteZ, input.ButtonEStop,
		input.ButtonEast, input.ButtonLT, input.ButtonLT2, input.ButtonLThumb, input.ButtonMenu, input.ButtonNorth,
		input.ButtonRT, input.ButtonRT2, input.ButtonRThumb, input.ButtonRecord, input.ButtonSelect,
		input.ButtonSouth, input.ButtonStart, input.ButtonWest, input.AbsolutePedalAccelerator,
		input.AbsolutePedalBrake, input.AbsolutePedalClutch:
		fallthrough
	default:
	}

	return linear, angular
}

func similar(a, b r3.Vector, deltaThreshold float64) bool {
	if math.Abs(a.X-b.X) > deltaThreshold {
		return false
	}

	if math.Abs(a.Y-b.Y) > deltaThreshold {
		return false
	}

	if math.Abs(a.Z-b.Z) > deltaThreshold {
		return false
	}

	return true
}

func scaleThrottle(a float64) float64 {
	//nolint:ifshort
	neg := a < 0

	a = math.Abs(a)
	if a <= .27 {
		return 0
	}

	a = math.Ceil(a*10) / 10.0

	if neg {
		a *= -1
	}

	return a
}

type throttleState struct {
	mu                              sync.Mutex
	linearThrottle, angularThrottle r3.Vector
	buttons                         map[input.Control]bool
	arrows                          map[input.Control]float64
}

func (ts *throttleState) init() {
	ts.buttons = map[input.Control]bool{
		input.ButtonNorth: false,
		input.ButtonSouth: false,
		input.ButtonEast:  false,
		input.ButtonWest:  false,
	}

	ts.arrows = map[input.Control]float64{
		input.AbsoluteHat0X: 0.0,
		input.AbsoluteHat0Y: 0.0,
	}
}

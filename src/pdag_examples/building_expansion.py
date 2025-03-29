from typing import Annotated, ClassVar, Literal, get_args, Mapping
import pdag

TPolicy = Literal[
    "do-nothing", "build-opt-57", "build-opt-33-and-rebuild", "build-exp-33-and-expand"
]
TAction = Literal[
    "do-nothing",
    "build-opt-33",
    "build-opt-57",
    "tear-down-and-build-opt-57",
    "expand",
    "build-exp-33",
]
TBuildingState = Literal["not-built", "opt-33", "opt-57", "exp-33", "exp-57"]


class DemandModel(pdag.Model):
    demand = pdag.RealParameter("demand", is_time_series=True)
    start_time = pdag.RealParameter("start_time")
    ramp_up_time = pdag.RealParameter("ramp_up_time")
    demand_max = pdag.RealParameter("demand_max")

    @pdag.relationship
    @staticmethod
    def demand_model(
        start_time: Annotated[float, start_time.ref()],
        ramp_up_time: Annotated[float, ramp_up_time.ref()],
        demand_max: Annotated[float, demand_max.ref()],
        n_time_steps: Annotated[int, pdag.ExecInfo("n_time_steps")],
    ) -> Annotated[list[float], demand.ref(all_time_steps=True)]:
        demand: list[float] = []
        for t in range(n_time_steps):
            if t < start_time:
                demand.append(0)
            elif t < start_time + ramp_up_time:
                demand.append(demand_max * (t - start_time) / ramp_up_time)
            else:
                demand.append(demand_max)
        return demand


class BuildingExpansionModel(pdag.Model):
    # Enumeration of possible values for policies, actions, and building states
    POLICIES: ClassVar[tuple[TPolicy, ...]] = get_args(TPolicy)
    ACTIONS: ClassVar[tuple[TAction, ...]] = get_args(TAction)
    BUILDING_STATES: ClassVar[tuple[TBuildingState, ...]] = get_args(TBuildingState)

    # Exogenous parameters
    demand_start_time = pdag.RealParameter(
        "demand_start_time",
        lower_bound=0,
        upper_bound=10,
        metadata={"XLRM": "X"},
    )
    demand_ramp_up_time = pdag.RealParameter(
        "demand_ramp_up_time",
        lower_bound=0,
        upper_bound=10,
        metadata={"XLRM": "X"},
    )
    demand_max = pdag.RealParameter(
        "demand_max",
        lower_bound=0,
        upper_bound=100,
        metadata={"XLRM": "X"},
    )
    # Demand is calculated in the DemandModel class
    demand = pdag.RealParameter(
        "demand",
        unit="floors",
        is_time_series=True,
    )
    revenue_per_floor = pdag.RealParameter(
        "revenue_per_floor",
        unit="USD/year/floor",
        lower_bound=600e3,
        upper_bound=1.5e6,
        metadata={"XLRM": "X"},
    )
    discount_rate = pdag.RealParameter(
        "discount_rate", lower_bound=0, upper_bound=0.1, metadata={"XLRM": "X"}
    )
    action_cost = pdag.Mapping(
        "action_cost",
        {
            "do-nothing": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=0,
                upper_bound=1,
                metadata={"XLRM": "X"},
            ),
            "build-opt-33": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=150e6,
                upper_bound=200e6,
                metadata={"XLRM": "X"},
            ),
            "build-opt-57": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=300e6,
                upper_bound=350e6,
                metadata={"XLRM": "X"},
            ),
            "tear-down": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=6e6,
                upper_bound=11e6,
                metadata={"XLRM": "X"},
            ),
            "expand": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=50e6,
                upper_bound=100e6,
                metadata={"XLRM": "X"},
            ),
            "build-exp-33": pdag.RealParameter(
                ...,
                unit="USD",
                lower_bound=180e6,
                upper_bound=220e6,
                metadata={"XLRM": "X"},
            ),
        },
    )

    # Decision parameters
    policy = pdag.CategoricalParameter(
        "policy",
        POLICIES,
        metadata={"XLRM": "L"},
    )
    rebuild_threshold = pdag.RealParameter(
        "rebuild_threshold",
        unit="floors",
        lower_bound=10,
        upper_bound=60,
        metadata={"XLRM": "L"},
    )
    expand_threshold = pdag.RealParameter(
        "expand_threshold",
        unit="floors",
        lower_bound=10,
        upper_bound=60,
        metadata={"XLRM": "L"},
    )

    # Calculated parameters
    action = pdag.CategoricalParameter(
        "action",
        ACTIONS,
        is_time_series=True,
    )
    building_state = pdag.CategoricalParameter(
        "building_state",
        BUILDING_STATES,
        is_time_series=True,
    )

    revenue = pdag.RealParameter("revenue", is_time_series=True)
    cost = pdag.RealParameter("cost", is_time_series=True)
    npv = pdag.RealParameter("npv")

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def determine_action(
        *,
        policy: Annotated[TPolicy, policy.ref()],
        building_state: Annotated[TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        rebuild_threshold: Annotated[float, rebuild_threshold.ref()],
        expand_threshold: Annotated[float, expand_threshold.ref()],
    ) -> Annotated[TAction, action.ref()]:
        match policy:
            case "do-nothing":
                return "do-nothing"
            case "build-opt-57":
                if building_state == "not-built":
                    return "build-opt-57"
                else:
                    return "do-nothing"
            case "build-opt-33-and-rebuild":
                if building_state == "not-built":
                    return "build-opt-33"
                elif building_state == "opt-33" and demand > rebuild_threshold:
                    return "tear-down-and-build-opt-57"
                else:
                    return "do-nothing"
            case "build-exp-33-and-expand":
                if building_state == "not-built":
                    return "build-exp-33"
                elif building_state == "exp-33" and demand > expand_threshold:
                    return "expand"
                else:
                    return "do-nothing"
            case _:
                raise ValueError(f"Unknown policy: {policy}")

    @pdag.relationship
    @staticmethod
    def initial_state() -> Annotated[TBuildingState, building_state.ref(initial=True)]:
        return "not-built"

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def state_transition_model(
        *,
        building_state: Annotated[TBuildingState, building_state.ref()],
        action: Annotated[TAction, action.ref()],
    ) -> Annotated[TBuildingState, building_state.ref(next=True)]:
        match action:
            case "do-nothing":
                return building_state
            case "build-opt-33":
                if building_state == "not-built":
                    return "opt-33"
            case "build-opt-57":
                if building_state == "not-built":
                    return "opt-57"
            case "build-exp-33":
                if building_state == "not-built":
                    return "exp-33"
            case "tear-down-and-build-opt-57":
                if building_state == "opt-33":
                    return "opt-57"
            case "expand":
                if building_state == "exp-33":
                    return "exp-57"
            case _:
                raise ValueError(f"Unknown action: {action}")

        raise ValueError(
            f"Invalid action '{action}' for building state '{building_state}'"
        )

    demand_model = DemandModel.to_relationship(
        "demand_model",
        inputs={
            DemandModel.start_time.ref(): demand_start_time.ref(),
            DemandModel.ramp_up_time.ref(): demand_ramp_up_time.ref(),
            DemandModel.demand_max.ref(): demand_max.ref(),
        },
        outputs={
            DemandModel.demand.ref(all_time_steps=True): demand.ref(
                all_time_steps=True
            ),
        },
    )

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_revenue(
        *,
        building_state: Annotated[TBuildingState, building_state.ref()],
        demand: Annotated[float, demand.ref()],
        revenue_per_floor: Annotated[float, revenue_per_floor.ref()],
    ) -> Annotated[float, revenue.ref()]:
        match building_state:
            case "not-built":
                capacity = 0
            case "opt-33":
                capacity = 33
            case "opt-57":
                capacity = 57
            case "exp-33":
                capacity = 33
            case "exp-57":
                capacity = 57
            case _:
                raise ValueError(f"Unknown building state: {building_state}")

        sales = min(capacity, demand)
        return sales * revenue_per_floor

    @pdag.relationship(at_each_time_step=True)
    @staticmethod
    def calculate_cost(
        *,
        action: Annotated[TAction, action.ref()],
        action_cost: Annotated[Mapping[TAction, float], action_cost.ref()],
    ) -> Annotated[float, cost.ref()]:
        if action == "tear-down-and-build-opt-57":
            return action_cost["tear-down"] + action_cost["build-opt-57"]
        return action_cost[action]

    @pdag.relationship
    @staticmethod
    def npv_calculation(
        *,
        revenue: Annotated[list[float], revenue.ref(all_time_steps=True)],
        cost: Annotated[list[float], cost.ref(all_time_steps=True)],
        discount_rate: Annotated[float, discount_rate.ref()],
    ) -> Annotated[float, npv.ref()]:
        return sum(
            (r - c) / ((1 + discount_rate) ** t)
            for t, (r, c) in enumerate(zip(revenue, cost, strict=True))
        )

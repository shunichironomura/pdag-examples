# mypy: disable-error-code="no-untyped-def, no-untyped-call"

import marimo

__generated_with = "0.11.31"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    import pdag
    import polars as pl

    from pdag_examples.building_expansion import BuildingExpansionModel
    return BuildingExpansionModel, Path, pd, pdag, pl


@app.cell
def _(Path, mo):
    import tomllib

    _capsula_config_path = mo.notebook_dir().parent / "capsula.toml"

    with _capsula_config_path.open("rb") as _f:
        _capsula_config = tomllib.load(_f)

    print(_capsula_config)

    vault_dir = Path(_capsula_config["vault-dir"])
    assert vault_dir.exists()
    return tomllib, vault_dir


@app.cell
def _(mo, vault_dir):
    file_browser = mo.ui.file_browser(
        initial_path=vault_dir,
        selection_mode="directory",
        multiple=False,
        restrict_navigation=True,
    )
    mo.vstack([mo.md("Select a results directory: "), file_browser])
    return (file_browser,)


@app.cell
def _(Path, file_browser, mo):
    mo.stop(
        file_browser.path() is None,
        mo.md("**Please select a results directory to continue**"),
    )
    results_dir = Path(file_browser.path())
    mo.md(f"Selected results directory: {results_dir}")
    return (results_dir,)


@app.cell
def _(results_dir):
    # Load pre-run report
    import json

    pre_run_report_path = results_dir / "pre-run-report.json"
    with pre_run_report_path.open() as f:
        pre_run_report = json.load(f)

    args = pre_run_report["function"]["run_experiments"]["bound_args"]
    args
    return args, f, json, pre_run_report, pre_run_report_path


@app.cell
def _(calc_distance, np, pd, results_dir, scenario_parameters):
    results_file_path = results_dir / "results.parquet"

    results = pd.read_parquet(results_file_path)

    # Calculate the "distance" from the nominal scenario
    hou = np.sqrt(
        sum(
            np.square(calc_distance(results[parameter_id.parameter_path_str], parameter))
            for parameter_id, parameter in scenario_parameters.items()
            if parameter.lower_bound != parameter.upper_bound
        ),
    )

    results["HoU"] = hou
    return hou, results, results_file_path


@app.cell
def _(BuildingExpansionModel, args, pdag):
    core_model = BuildingExpansionModel.to_core_model()
    exec_model = pdag.create_exec_model_from_core_model(core_model=core_model, n_time_steps=args["n_time_steps"])
    return core_model, exec_model


@app.cell
def _(scenario_parameters):
    sorted([param_id.parameter_path_str for param_id in scenario_parameters])
    return


@app.cell
def _(exec_model):
    input_parameters = exec_model.input_parameters()
    scenario_parameters = {
        parameter_id: parameter
        for parameter_id, parameter in input_parameters.items()
        if parameter.metadata.get("XLRM", None) == "X"
    }
    decision_parameters = {
        parameter_id: parameter
        for parameter_id, parameter in input_parameters.items()
        if parameter.metadata.get("XLRM", None) == "L"
    }

    return decision_parameters, input_parameters, scenario_parameters


@app.cell
def _(
    decision_parameters,
    exec_model,
    input_parameters,
    pdag,
    results,
    scenario_parameters,
):

    static_input_parameter_ids = [param_id for param_id in input_parameters if isinstance(param_id, pdag.StaticParameterId)]
    varied_static_input_parameter_ids = [
        param_id
        for param_id in static_input_parameter_ids
        if results[param_id.parameter_path_str].nunique() > 1  # noqa: PD101
    ]

    outcome_parameter_ids = set(exec_model.parameter_ids) - set(scenario_parameters) - set(decision_parameters)
    static_outcome_parameter_ids = [
        param_id for param_id in outcome_parameter_ids if isinstance(param_id, pdag.StaticParameterId)
    ]
    varied_static_outcome_parameter_ids = [
        param_id
        for param_id in static_outcome_parameter_ids
        if results[param_id.parameter_path_str].nunique() > 1  # noqa: PD101
    ]
    return (
        outcome_parameter_ids,
        static_input_parameter_ids,
        static_outcome_parameter_ids,
        varied_static_input_parameter_ids,
        varied_static_outcome_parameter_ids,
    )


@app.cell
def _(scenario_parameters):
    [parameter_id.parameter_path_str for parameter_id in scenario_parameters]
    return


@app.cell
def _(pdag, scenario_parameters):
    import numpy as np

    _parameter_id = next(iter(scenario_parameters))
    print(_parameter_id)

    _parameter = scenario_parameters[_parameter_id]


    def calc_distance(x, parameter):
        assert isinstance(parameter, pdag.RealParameter)
        _nominal = (parameter.lower_bound + parameter.upper_bound) / 2
        _half_range = (parameter.upper_bound - parameter.lower_bound) / 2
        return np.abs((x - _nominal) / _half_range)


    print(_parameter)

    _nominal = (_parameter.lower_bound + _parameter.upper_bound) / 2
    print(_nominal)

    _half_range = (_parameter.upper_bound - _parameter.lower_bound) / 2


    return calc_distance, np


@app.cell
def _(np, results):
    np.abs((results[_parameter_id.parameter_path_str] - _nominal) / _half_range)
    return


@app.cell
def _(calc_distance, results, scenario_parameters):
    for _parameter_id, _parameter in scenario_parameters.items():
        if _parameter.lower_bound == _parameter.upper_bound:
            print(f"Skipping parameter {_parameter.name} with fixed value")
            continue
        _distance = calc_distance(results[_parameter_id.parameter_path_str].to_numpy(), _parameter)
        # print(_distance)
    return


@app.cell
def _(BuildingExpansionModel, static_results):
    static_results.plot.scatter(x="HoU", y=BuildingExpansionModel.npv.name)
    return


@app.cell
def _(static_results):
    decision_ids = static_results["decision_id"].unique()
    return (decision_ids,)


@app.cell
def _(BuildingExpansionModel):
    outcome_cols = [
        BuildingExpansionModel.npv.name,
    ]
    outcome_cols
    return (outcome_cols,)


@app.cell
def _(decision_parameters, results):
    decisions = {}
    for _decision_id, _group in results.groupby("decision_id"):
        _first_row = _group.iloc[0].to_dict()
        decisions[_decision_id] = {
            param_id.parameter_path_str: _first_row[param_id.parameter_path_str] for param_id in decision_parameters
        }
    decisions
    return (decisions,)


@app.cell
def _(decision_ids, decisions, mo, np, outcome_cols, pd, static_results):
    # _horizons = np.linspace(0, 1, 100)

    # _dfs = []
    # for _decision_id in mo.status.progress_bar(decision_ids[:2]):
    #     for _i, _horizon in enumerate(_horizons):
    #         _results_within_horizon_decision = static_results[
    #             (static_results["HoU"] < _horizon) & (static_results["decision_id"] == _decision_id)
    #         ]
    #         _npv_min = _results_within_horizon_decision[BuildingExpansionModel.npv.name].min()
    #         _npv_max = _results_within_horizon_decision[BuildingExpansionModel.npv.name].max()
    #         _dfs.append(
    #             pd.DataFrame(
    #                 {
    #                     "HoU": _horizon,
    #                     "NPVMin": _npv_min,
    #                     "NPVMax": _npv_max,
    #                     "DecisionId": _decision_id,
    #                 },
    #                 index=[_i],
    #             )
    #         )
    # df_hou = pd.concat(_dfs, ignore_index=True)

    # Flexible horizon points; adjust number as needed.
    horizons = np.linspace(0, 1, 101)

    _dfs = []
    for _decision_id in mo.status.progress_bar(decision_ids):
        # Filter once for the current decision.
        df_dec = static_results[static_results["decision_id"] == _decision_id]
        if df_dec.empty:
            continue

        # Sort by HoU; this makes the cumulative operations efficient.
        df_dec_sorted = df_dec.sort_values("HoU")
        hou_values = df_dec_sorted["HoU"].values

        # Use vectorized search to find, for each horizon, the last index where HoU < horizon.
        # np.searchsorted returns the index where each horizon should be inserted.
        indices = np.searchsorted(hou_values, horizons, side="right") - 1

        _row = {
            "HoU": horizons,
            # "NPVMin": npv_min_values,
            # "NPVMax": npv_max_values,
            "DecisionId": _decision_id,
        }
        for _outcome_col in outcome_cols:
            # Get the npv column values.
            _values = df_dec_sorted[_outcome_col].values

            # Compute cumulative min and max.
            _cum_min = np.minimum.accumulate(_values)
            _cum_max = np.maximum.accumulate(_values)

            # For horizon values below the smallest HoU, assign np.nan.
            _min_values = np.where(indices >= 0, _cum_min[indices], np.nan)
            _max_values = np.where(indices >= 0, _cum_max[indices], np.nan)

            _row[f"{_outcome_col}-min"] = _min_values
            _row[f"{_outcome_col}-max"] = _max_values

        _row = _row | decisions[_decision_id]

        # Build the DataFrame for this decision.
        df_temp = pd.DataFrame(_row)
        _dfs.append(df_temp)

    # Concatenate results from all decisions.
    df_hou = pd.concat(_dfs, ignore_index=True)

    # df_hou.to_parquet(results_dir / "hou.parquet")
    return (
        df_dec,
        df_dec_sorted,
        df_hou,
        df_temp,
        horizons,
        hou_values,
        indices,
    )


@app.cell
def _(df_hou, results_dir):
    df_hou.to_parquet(results_dir / "hou.parquet")
    return


@app.cell
def _(static_results):
    static_results
    return


@app.cell
def _(df_hou):
    df_hou
    return


@app.cell
def _(df_hou):
    df_hou.columns
    return


@app.cell
def _(df_hou):
    import seaborn as sns
    import matplotlib.pyplot as plt

    _fig, _axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    _ax = _axes[0]
    sns.lineplot(
        data=df_hou,
        x="HoU",
        y="npv-max",
        hue="policy",
        # palette="tab20",
        alpha=0.5,
        legend=True,
        ax=_ax,
        linewidth=1,
    )
    _ax.set_title("NPVMax vs HoU by policy")
    _ax.set_xlabel("HoU")
    _ax.set_ylabel("NPVMax")
    _ax.grid()

    _ax = _axes[1]
    sns.lineplot(
        data=df_hou,
        x="HoU",
        y="npv-min",
        hue="policy",
        # palette="tab20",
        alpha=0.5,
        legend=True,
        ax=_ax,
        linewidth=1,
    )
    _ax.set_title("NPVMin vs HoU by policy")
    _ax.set_xlabel("HoU")
    _ax.set_ylabel("NPVMin")
    _ax.grid()

    plt.show()
    return plt, sns


@app.cell
def _(df_hou, horizons, sns):
    # Minimum NPV vs Maximum NPV trade-off

    _horizon = horizons[50]
    _hou_at_horizon = df_hou[df_hou["HoU"] == _horizon]
    sns.scatterplot(data=df_hou, x="npv-min", y="npv-max", hue="HoU", size="HoU", sizes=(300, 30))
    return


@app.cell
def _(decision_parameters):
    decision_parameters
    return


@app.cell
def _(decision_parameters, mo):
    npv_tradeoff_plot_color_selector = mo.ui.radio(options=[param.parameter_path_str for param in decision_parameters])
    npv_tradeoff_plot_color_selector
    return (npv_tradeoff_plot_color_selector,)


@app.cell
def _(decision_parameters, df_hou, horizons, np, sns):
    _horizon = horizons[78]

    _df_hou_at_h = df_hou[df_hou["HoU"] == _horizon].copy()
    print(_df_hou_at_h.columns)
    _decision_parameter_paths = [param.parameter_path_str for param in decision_parameters]
    _df_hou_at_h["npv-min"].max()
    _df_hou_at_h["npv-min_log"] = np.log10(-_df_hou_at_h["npv-min"] + 1)
    _cols = [col for col in _decision_parameter_paths if not col.startswith("otv_arch_selection_order")] + ["npv-min_log"]
    sns.pairplot(
        _df_hou_at_h,
        x_vars=_cols,
        y_vars=_cols,
        hue="npv-min_log",
        palette="viridis_r",
        plot_kws={"alpha": 0.5},
    )
    return


@app.cell
def _(decision_parameters, df_hou, horizons, np, sns):
    _horizon = horizons[78]

    df_hou_at_h = df_hou[df_hou["HoU"] == _horizon].copy()
    print(df_hou_at_h.columns)
    _decision_parameter_paths = [param.parameter_path_str for param in decision_parameters]
    df_hou_at_h["npv-min"].max()
    df_hou_at_h["npv-min_log"] = np.log10(-df_hou_at_h["npv-min"] + 1)
    sns.pairplot(
        df_hou_at_h,
        x_vars=[col for col in _decision_parameter_paths if not col.startswith("otv_arch_selection_order")],
        y_vars=[
            col
            for col in df_hou_at_h.columns
            if col not in _decision_parameter_paths and col != "DecisionId" and col != "HoU"
        ],
        hue="npv-min_log",
        palette="viridis_r",
        plot_kws={"alpha": 0.5},
    )
    return (df_hou_at_h,)


@app.cell
def _(
    BuildingExpansionModel,
    decision_parameters,
    scenario_parameters,
    static_results,
):
    from ema_workbench.analysis import prim
    from itertools import chain

    _experiments_cols = [param.parameter_path_str for param in chain(decision_parameters, scenario_parameters)]
    print(sorted(_experiments_cols))
    assert set(_experiments_cols) <= set(static_results.columns)
    _outcome_col = BuildingExpansionModel.npv.name

    _x = static_results[_experiments_cols]
    _y = static_results[_outcome_col] >= 0
    _prim_alg = prim.Prim(_x, _y, threshold=0.8)
    box1 = _prim_alg.find_box()
    return box1, chain, prim


@app.cell
def _(static_results):
    static_results
    return


@app.cell
def _(decision_parameters, df_hou, npv_tradeoff_plot_color_selector):
    import plotly.express as px

    fig = px.scatter(
        df_hou,
        x="npv-min",
        y="npv-max",
        color=npv_tradeoff_plot_color_selector.value,
        hover_data=["DecisionId"] + [param.parameter_path_str for param in decision_parameters],
        animation_frame="HoU",
        title="NPVMin vs NPVMax with Inverted Size Mapping Based on HoU",
    )
    fig.show()
    return fig, px


@app.cell
def _(df_hou, plt, results, sns):
    selected_decision_id = "T9nfOV"
    results_decision = results[results["decision_id"] == selected_decision_id]
    df_hou_decision = df_hou[df_hou["DecisionId"] == selected_decision_id]

    _fig, _axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    _ax = _axes[0]
    sns.lineplot(data=df_hou_decision, x="HoU", y="npv-max", ax=_ax)
    _ax.set_title(f"NPVMax vs HoU by for decision ID {selected_decision_id}")
    _ax.set_xlabel("HoU")
    _ax.set_ylabel("NPV Max")
    _ax.grid()

    _ax = _axes[1]
    sns.lineplot(data=df_hou_decision, x="HoU", y="npv-min", ax=_ax)
    _ax.set_title(f"NPV Min vs HoU by for decision ID {selected_decision_id}")
    _ax.set_xlabel("HoU")
    _ax.set_ylabel("NPV Min")
    _ax.grid()

    plt.show()

    results_decision
    return df_hou_decision, results_decision, selected_decision_id


@app.cell
def _(decision_parameters, results_decision):
    _first_row_dict = results_decision.iloc[0].to_dict()
    _dp = next(iter(decision_parameters))
    _dp.parameter_path_str

    selected_decision = {param_id: _first_row_dict[param_id.parameter_path_str] for param_id in decision_parameters}
    selected_decision
    return (selected_decision,)


@app.cell
def _(results):
    results["HoU"].hist()
    return


@app.cell
def _(BuildingExpansionModel, results):
    results[BuildingExpansionModel.npv.name].hist()
    return


@app.cell
def _(BuildingExpansionModel, mo, results):
    npv_min = results[BuildingExpansionModel.npv.name].min()
    npv_max = results[BuildingExpansionModel.npv.name].max()

    mo.md(f"**NPV range: {npv_min:.2e} to {npv_max:.2e}**")
    return npv_max, npv_min


@app.cell
def _(BuildingExpansionModel, results):
    results[results[BuildingExpansionModel.npv.name] < -1e12]
    return


@app.cell
def _(results):
    static_results = results[results["time_step"] == 0]
    return (static_results,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

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
    return args, f, json, pre_run_report, pre_run_report_path


@app.cell
def _(args, mo):
    mo.md(f"""\
    ## Summary of results

    Arguments for the selected run

    - Number of time steps: {args["n_time_steps"]}
    - Number of scenarios: {args["n_scenarios"]}
    - Number of decisions: {args["n_decisions"]}
    - Number of time steps: {args["n_time_steps"]}
    """)
    return


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
def _(decision_parameters, mo, scenario_parameters):
    _sorted_scenario_params = sorted([param_id.parameter_path_str for param_id in scenario_parameters])
    _sorted_decision_params = sorted([param_id.parameter_path_str for param_id in decision_parameters])

    mo.md(f"""\
    Scenario parameters:

    {'\n'.join(f'- `{param}`' for param in _sorted_scenario_params)}'

    Decision parameters:

    {'\n'.join(f'- `{param}`' for param in _sorted_decision_params)}'
    """)
    return


@app.cell
def _(mo):
    mo.md("""Distribution of NPV values""")
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
def _(mo):
    mo.md(r"""## Horizon of Uncertainty (HoU) plot""")
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
def _(BuildingExpansionModel, sns, static_results):
    sns.set_theme()
    sns.scatterplot(
        data=static_results,
        x="HoU",
        y=BuildingExpansionModel.npv.name,
        hue="policy",
        marker="+",
        alpha=0.5,
        legend=True,
    )
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
    return (outcome_cols,)


@app.cell
def _(decision_parameters, results):
    decisions = {}
    for _decision_id, _group in results.groupby("decision_id"):
        _first_row = _group.iloc[0].to_dict()
        decisions[_decision_id] = {
            param_id.parameter_path_str: _first_row[param_id.parameter_path_str] for param_id in decision_parameters
        }
    return (decisions,)


@app.cell
def _(decision_ids, decisions, mo, np, outcome_cols, pd, static_results):
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
def _(df_hou, mo):
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

    mo.mpl.interactive(plt.gcf())
    return plt, sns


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

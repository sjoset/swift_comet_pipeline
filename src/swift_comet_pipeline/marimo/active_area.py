import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pathlib
    return mo, pathlib


@app.cell
def _():
    from swift_comet_pipeline.pipeline.project_configuration.read_comet_project_config import read_comet_project_config
    return (read_comet_project_config,)


@app.cell
def _(mo, pathlib):
    file_browser = mo.ui.file_browser(
        initial_path=pathlib.Path("~").expanduser(), selection_mode='file'
    )
    mo.vstack([mo.ui.file_browser()])
    return (file_browser,)


@app.cell
def _(file_browser):
    config_path = file_browser.path()
    config_path
    return


@app.cell
def _(file_browser):
    file_browser.value
    return


@app.cell
def _(file_browser, read_comet_project_config):
    pc = read_comet_project_config(file_browser.path(index=0))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

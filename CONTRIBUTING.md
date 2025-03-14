## Contributing In General

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/docling-project/docling-sdg/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/docling-project/docling-sdg/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

For general questions or support requests, please refer to the [discussion section](https://github.com/docling-project/docling-sdg/discussions).

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/docling-project/docling-sdg/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/docling-project/docling-sdg/issues) before sending a
pull request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from two of the
maintainers of each component affected.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.


## Legal

Each source file must include a license header for the MIT
Software. Using the SPDX format is the simplest approach.
e.g.

```
/*
Copyright IBM Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/
```

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```
git commit -s
```


## Communication

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling-sdg/discussions).


## Setup

### Usage of Ruff

We use [Ruff](https://docs.astral.sh/ruff/) to manage dependencies.

#### Installation

To install, follow the documentation here: https://docs.astral.sh/uv/getting-started/installation/

1. Install the `uv` globally on your machine:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. The official guidelines linked above include useful details on configuring autocomplete for most shell environments, e.g., Bash and Zsh.


#### Create a Virtual Environment and Install Dependencies

To create the virtual environment, run:

```bash
uv venv
```

The virtual environment can be _activated_ to make its packages available:

```bash
source .venv/bin/activate
```

Then, to install dependencies, run:

```bash
uv sync
```


**(Advanced) Use a Specific Python Version**

If you need to work with a specific (older) version of Python, for instance, `3.11`, run:

```bash
uv venv --python 3.11
```

More detailed options are described in the [uv documentation](https://docs.astral.sh/uv/pip/environments).


#### Add a New Dependency

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```



## Code Sytle Guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), as linter and code formatter
- [MyPy](https://mypy.readthedocs.io), to check typing specs

We run a series of checks on every commit, including a test suite, using [pre-commit](https://pre-commit.com/).
To install the hooks, run:

```bash
uv run pre-commit install
```

To run the checks on-demand, run:

```bash
uv run pre-commit run --all-files
```

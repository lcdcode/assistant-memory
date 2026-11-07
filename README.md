# assistant-memory

A local-only, persistent memory MCP server for Claude Code (and any MCP client).
Memories are embedded with a local Ollama model, stored in PostgreSQL, and retrieved
with hybrid semantic + lexical search re-ranked by an ACT-R cognitive activation model.
Nothing leaves the machine: no cloud APIs, no tokens spent on embeddings.

This is a **lite, headless variant** - the MCP server only, with no web UI.

## Attribution

This project is a lite reimplementation inspired by
[MrZzE00/MCP-Claude-mem-local](https://github.com/MrZzE00/MCP-Claude-mem-local) (MIT licensed).

- The core ideas are shared: local Postgres + pgvector storage, Ollama embeddings,
  ACT-R cognitive scoring, hybrid retrieval, and strategic forgetting.
- This variant is **headless**. Upstream ships a web interface for browsing, searching,
  and filtering memories; this version exposes those capabilities only through MCP tools.
- This is an independent implementation rather than a fork, so the source code, schema,
  and tunable parameters differ. It was assembled from first principles against the same
  cognitive-science and IR foundations (see Design below).

## Features

- **Six MCP tools**: `store_memory`, `retrieve_memories`, `list_memories`,
  `delete_memory`, `memory_stats`, `memory_forgetting_cycle`.
- **Local embeddings** via Ollama (`nomic-embed-text`, 768-dim). The Ollama host is
  pinned to localhost in `server.py` as an SSRF guard.
- **Hybrid retrieval**: pgvector cosine similarity fused with pg_trgm trigram matching
  via Reciprocal Rank Fusion (k=60). Trigram failures degrade gracefully to vector-only.
- **ACT-R cognitive re-ranking**: results are scored by an activation function combining
  base-level decay (recency + frequency of access), semantic similarity, and optional
  associative spreading from query tags, plus noise. Fully tunable via `.env`.
- **Strategic forgetting**: `memory_forgetting_cycle` recomputes activation for every
  memory and sets its status - `active` (A>0), `dormant` (-2<A<=0), or `forgotten` (A<=-2).
  Forgotten rows stay in the database but are excluded from default queries.
- **Categories** (CHECK-constrained): `bugfix`, `decision`, `feature`, `discovery`,
  `refactor`, `change`, `learning`, `pattern`, `error_solution`, `preference`.
- **Filtering** by category, tags, project context, and minimum similarity.
- **Per-user scoping** via `USER_ID`, plus access-count and access-timestamp tracking.

## Design

The activation score is the ACT-R declarative-memory equation:

```
A(m) = B(m) + w * cos_sim(query, m) + S(m) + epsilon
```

- `B(m)` - base-level activation from access recency and frequency (decay `d`).
- `w * cos_sim` - semantic relevance weight.
- `S(m)` - associative spreading from shared tags (optional).
- `epsilon` - logistic noise (optional).

Retrieval prefetches a wider candidate set (`ACTR_PREFETCH_LIMIT`), fuses the vector and
trigram rankings with RRF, then ACT-R re-ranks and truncates to `max_results`. Set
`USE_ACTR_SCORING=false` to fall back to classic `cos_sim * importance` ordering.

## Requirements

- PostgreSQL >= 13 (>= 14 recommended) with the `pgvector` (>= 0.5.0) and `pg_trgm` extensions.
- [Ollama](https://ollama.com) running locally with an embedding model pulled:
  `ollama pull nomic-embed-text`.
- Python 3.11+ (the project uses a `venv`). The lock file is resolved on 3.11.

### Dependencies

`requirements.in` lists the direct dependencies with version ranges. `requirements.txt` is the
generated lock file (exact versions plus hashes) that both Debian and Termux install from. To
change a dependency, edit `requirements.in` and regenerate the lock with
[pip-tools](https://pip-tools.readthedocs.io):

```sh
pip-compile --generate-hashes --strip-extras --allow-unsafe -o requirements.txt requirements.in
```

Commit both files together, then run the install command below on each machine.

## Setup

```sh
# 1. Python dependencies
python3 -m venv venv
./venv/bin/pip install --require-hashes -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env: set PG_HOST, PG_USER and USER_ID at minimum (see Authentication below)

# 3. Pull the embedding model
ollama pull nomic-embed-text

# 4. Create the database and apply the schema (idempotent)
scripts/init_db.sh
```

### First-time bootstrap on a fresh install

On a clean PostgreSQL install, `scripts/init_db.sh` trips over two things before it can
apply the schema. Both are one-time host setup, not bugs in the script.

#### 1. Authentication: connect over the Unix socket

Point `PG_HOST` at PostgreSQL's Unix socket directory instead of `127.0.0.1`. Socket
connections are checked against the `local` rules in `pg_hba.conf`, which on Debian default
to `peer`: your OS login is the credential, so no password and no `pg_hba.conf` edit.

| Platform | Socket directory (`PG_HOST`) |
| --- | --- |
| Debian / Ubuntu | `/var/run/postgresql` |
| Termux | `/data/data/com.termux/files/usr/tmp` (`$PREFIX/tmp`) |

A fresh cluster only ships the `postgres` superuser, so create a role named after your
login (`peer` matches the two):

```sh
sudo -u postgres createuser --createdb --login "$(whoami)"
```

Then set `PG_HOST` and `PG_USER` in `.env` and leave `PG_PASSWORD` empty.

If you need TCP (`PG_HOST=127.0.0.1`), keep the default `scram-sha-256` method on the
`host` lines, give your role a password (`\password` in `psql`), and set `PG_PASSWORD` in
`.env`. Do not switch the `host` lines to `trust`: anything that can reach the port can then
connect as any role, superuser included, with no credentials.

#### Termux / Android: turn off TCP

`initdb` without `--auth` (as on Termux) sets every `pg_hba.conf` line to `trust`, and
PostgreSQL listens on `localhost` by default. On Android, localhost is not private: the
loopback interface is shared by every app on the device, so any installed app with the
`INTERNET` permission can connect to `127.0.0.1:5432`. Under `trust` it logs in as your
role - the cluster superuser on Termux - and can read every memory and run shell commands
as your Termux user (`COPY ... TO PROGRAM`).

The socket lives in Termux's private `$PREFIX/tmp`, which other apps cannot enter. Disable
TCP in `$PGDATA/postgresql.conf`:

```
listen_addresses = ''
unix_socket_permissions = 0700
```

As a second layer, change the method on the `host` lines in `$PGDATA/pg_hba.conf` from
`trust` to `reject`. Then restart and set the socket directory in `.env`:

```sh
pg_ctl -D "$PGDATA" restart
# .env: PG_HOST=/data/data/com.termux/files/usr/tmp
```

The `local` lines can stay `trust`: only processes running as your Termux user can reach
the socket, and those can already read `$PGDATA` directly.

> Ollama has the same exposure on `127.0.0.1:11434` and cannot listen on a Unix socket. It
> holds no memories, but any app on the device can use it to run models or to pull and
> delete them.

#### 2. The `vector` extension needs a superuser

`sql/schema.sql` runs `CREATE EXTENSION vector`. `pg_trgm` is a trusted extension (a DB
owner can create it), but **pgvector is not trusted, so creating it requires a
superuser** - your `--createdb` role will fail on that line.

First make sure the pgvector package is installed for the cluster:

```sh
sudo apt install postgresql-$(pg_config --version | grep -oE '[0-9]+' | head -1)-pgvector
# (package name varies by distro; e.g. `pgvector` on Arch, build from source elsewhere)
```

Then have the superuser create the database and both extensions once. After that the
`CREATE EXTENSION IF NOT EXISTS` lines in the schema become no-ops and need no special
privilege:

```sh
sudo -u postgres createdb -O "$(whoami)" assistant_memory
sudo -u postgres psql -d assistant_memory \
   -c 'CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm;'

# Now the schema applies cleanly as your normal role
scripts/init_db.sh
```

Alternatively, for a throwaway local box, grant your role superuser and let the script do
everything itself: `sudo -u postgres psql -c 'ALTER ROLE "<user>" SUPERUSER;'`.

### Configuration (`.env`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `PG_HOST` / `PG_PORT` | `localhost` / `5432` | Unix socket directory (recommended) or TCP host. |
| `PG_DATABASE` | `assistant_memory` | Target database. |
| `PG_USER` / `PG_PASSWORD` | - | DB credentials (socket `peer` auth leaves the password empty). |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint (localhost-only, enforced). |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model; 768-dim must match the schema. |
| `USER_ID` | `default` | Owner tag for stored/retrieved memories. |
| `USE_ACTR_SCORING` | `true` | Toggle ACT-R re-ranking. |
| `ACTR_DECAY_D` | `0.5` | Base-level decay rate. |
| `ACTR_WEIGHT_W` | `11.0` | Semantic similarity weight. |
| `ACTR_NOISE_SIGMA` | `1.2` | Logistic noise scale. |
| `ACTR_THRESHOLD_TAU` | `-2.0` | Forgetting threshold. |
| `ACTR_SPREADING_S` | `2.0` | Associative spreading strength. |
| `ACTR_USE_SPREADING` / `ACTR_USE_NOISE` | `true` | Toggle those terms. |
| `ACTR_PREFETCH_LIMIT` | `50` | Candidate pool size before re-ranking. |

> If you change `EMBEDDING_MODEL` to one with a different dimension, update `VECTOR(768)`
> in `sql/schema.sql` and re-embed existing rows.

## Usage

Register the server with your MCP client. For Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "assistant-memory": {
      "type": "stdio",
      "command": "/absolute/path/to/assistant-memory/venv/bin/python",
      "args": ["/absolute/path/to/assistant-memory/src/server.py"],
      "env": {}
    }
  }
}
```

The server speaks MCP over stdio; the client launches it on demand. Make sure PostgreSQL
and Ollama are running first (see `run-assistant-example.sh` for an example launcher that starts
both, then execs `claude`).

### Tools

| Tool | Purpose |
| --- | --- |
| `store_memory(content, category, summary?, tags?, importance?, project?)` | Persist a memory; embeds the content. |
| `retrieve_memories(query, max_results?, category?, min_similarity?, tags?, project?, include_forgotten?)` | Hybrid + ACT-R semantic search. |
| `list_memories(limit?, category?, tags?, project?)` | List recent memories, newest first. |
| `delete_memory(memory_id)` | Delete one memory by UUID. |
| `memory_stats()` | Counts by category and status, most-accessed, average activation. |
| `memory_forgetting_cycle()` | Recompute activation and update active/dormant/forgotten status. |

## Project layout

```
src/
  server.py          MCP server, tool definitions, DB pool, Ollama embedding
  actr_scoring.py    ACT-R activation scoring and ranking
  hybrid_search.py   vector + trigram query builders, RRF fusion
  forgetting.py      strategic forgetting cycle (status transitions)
sql/schema.sql       idempotent DDL (table, extensions, indexes, CHECK constraints)
scripts/init_db.sh   create database + apply schema
.env.example         configuration template
```

## Notes and limitations

- **Memory rows do not travel with the repo.** A fresh clone starts with an empty
  `memories` table; the schema is portable, the data is not. Cross-machine sync
  (encrypted `pg_dump`, logical replication, or a shared private DB) is not yet solved.
- **No migration tooling.** If the schema drifts, add Alembic or numbered
  `sql/migrations/` files.
- Categories and `memory_status` are enforced with CHECK constraints (not ENUMs) so the
  schema stays in sync with `server.py` without DDL migrations.

## License

This lite variant follows the upstream MIT license. See the
[original project](https://github.com/MrZzE00/MCP-Claude-mem-local) for the canonical terms.

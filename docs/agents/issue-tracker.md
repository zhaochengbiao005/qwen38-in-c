# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files under `.wayfinder/`.

## Conventions

- One effort per map: the map is `.wayfinder/map.md`
- Tickets are one file each at `.wayfinder/tickets/NN-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each ticket file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under `## Comments`
- Research artifacts produced while resolving tickets go to `.wayfinder/research/` and are linked from the ticket, never pasted in

## When a skill says "publish to the issue tracker"

Create a new file under `.wayfinder/tickets/` (numbered next in sequence).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the ticket number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.wayfinder/map.md` (Destination / Notes / Decisions-so-far / Not yet specified / Out of scope).
- **Child ticket**: `.wayfinder/tickets/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Label:` line records the wayfinder type (`research`/`prototype`/`grilling`/`task`); `Claim:` records claimed/未认领; resolved tickets carry a `## Resolution` section and are noted as CLOSED.
- **Blocking**: a `Blocked by:` line near the top. A ticket is unblocked when every ticket it lists is closed.
- **Frontier**: scan `.wayfinder/tickets/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Claim: 已认领（<会话标识>）` and save before any work.
- **Resolve**: append the answer under `## Resolution`, mark CLOSED, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.

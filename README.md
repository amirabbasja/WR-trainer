# WR Trainer

The WR client follows the PearlHash exchange captured from WildRig Multi 0.52.3.
It starts each connection with login request ID `0` and defaults to the agent
`WildRig_0.52.3`. An explicitly configured custom agent still takes precedence.

The first `report_info` is due 15 seconds after login. Subsequent reports are due
120.6 seconds apart. The trainer sends a due report after completing a batch.

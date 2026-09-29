# Vendored third-party code

* `moflow/` - from [MoFlow](https://github.com/felix-yuxiang/MoFlow) (flow-matching
  trajectory forecaster): `FlowMatcher` base class, MTR decoder, MLP helpers.
* `ocsort/` - from [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)
  (`trackers/ocsort_tracker/`): Kalman box tracker and OC-SORT association.

Both are vendored unmodified except for import-path fixes, so the repository
runs out of the box. Original licenses apply to these directories.

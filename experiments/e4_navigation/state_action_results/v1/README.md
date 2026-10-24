# Direct state-action safety scorer baseline v1

This is an actually trained5059-parameter MLP, **not LeWM, not a Navigation world model or readout integration**. User confirmed the dedicated Navigation weights have not arrived. It receives only current observed state, relative goals and candidate action sequences (44 features). No simulator future or safety label enters inference.

## Frozen protocol and results

Train12 geometries/48 groups/3072 candidates; validation3 geometries/12 groups/768 candidates. Select epoch and threshold on validation; freeze weights and policy; activate test3 geometries/12 groups/768 candidates once. Three transformed route families remain shared; not held-out-family generalization.

Model44→64→32→3 has heads unsafe logit, scaled original soft cost and scaled minimum substep separation. CPU seed0, AdamW lr.001 wd.01, class-balanced BCE+0.1 cost MSE+separation MSE. Feature and target statistics use train only. Validation selection minimizes scene-average unsafe selection when a safe candidate exists, then original soft regret. Maximum200 epochs, patience30, thresholds0.1/0.25/0.5. Selected epoch126, threshold0.25.

| Split | Scoring rule | Unsafe when safe available* | Unsafe over all states* | Original soft regret* |
|---|---|---:|---:|---:|
| validation | learned_cost_only | 16.67% | 16.67% | 0.4917 |
| validation | learned_safety | 0.00% | 0.00% | 0.6453 |
| validation | true_soft_oracle | 91.67% | 91.67% | 0.0000 |
| validation | true_safety_oracle | 0.00% | 0.00% | 0.3572 |
| test | learned_cost_only | 16.67% | 25.00% | 0.6822 |
| test | learned_safety | 8.33% | 16.67% | 0.6987 |
| test | true_soft_oracle | 91.67% | 91.67% | 0.0000 |
| test | true_safety_oracle | 0.00% | 8.33% | 0.2801 |

*Average within geometry over eligible offsets, then equally across geometries. `learned_cost_only` uses the cost head of the SAME safety-selected network: a scoring-rule ablation, not an independently selected cost-only competitor. True soft/safety oracles use future supervision and are diagnostic only. A soft oracle can prefer unsafe actions because the original soft objective does not enforce safety.

Risk policy: among predicted-safe candidates choose lowest predicted cost; if all predicted unsafe, choose largest predicted minimum separation then lowest cost. This fallback is not guaranteed safe. Test includes an all-unsafe true candidate pool, so conditional and unconditional unsafe rates differ; true-safety oracle unconditional risk is not zero.

Test conditional unsafe reduction is8.33 percentage points in1/3 geometries. Predeclared success required>=10 points AND>=2/3 geometries: **not met**. Validation0% does not transfer perfectly; retain this negative/limited result. No closed-loop episode, Navigation checkpoint claim, LeWM improvement, or post-test refit is reported.

## Validation and storage

Sixteen targeted unit tests pass: prior physics checks plus feature contract, future-label argument rejection, pre-IO test guard, checkpoint-hash guard, fallback, safe selection and parameter count. Independent NumPy reconstruction verifies train-only statistics, all test input features, forward predictions (maximum absolute difference1.91e-06), every selected candidate/regret and the success criterion. Weight hash remains unchanged after test. Raw model/portable checkpoint and predictions remain external; frozen hashes identify them.

The previously sealed test split has now been consumed by this frozen experiment. It cannot be repeatedly used to select thresholds, models or features. Earlier conflict-data reports refer to its status at creation; this report and current PROJECT_STATE supersede that status. A later change needs new confirmation geometries.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/state_action_ranker.py --data outputs/conflict_ranking_v1 --output outputs/state_action_ranker_v1
python experiments/e4_navigation/verify_state_action_ranker.py --data outputs/conflict_ranking_v1 --run outputs/state_action_ranker_v1
```

Required next material: Navigation-specific checkpoint, load API/config/normalization, agent/history/action contract and compatible state head. This direct scorer remains a separately named baseline when those arrive. Do not plug PushT weights into the Navigation world-model adapter.

import copy
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from uav_transport.local_model import (FEATURE_NAMES, TARGET_NAMES, FrozenPredictor,
    features, fit_model, load_dataset, load_model, prediction_metrics, save_model)


def fixture():
    plant=SimpleNamespace(dt=.002,control_dt=.02,drone_mass=.034,payload_mass=.03,
        rotor_max=.16,motor_tau=.03,cable_length=.6,formation_radius=.25,gravity=9.81,
        arm_length=.046,yaw_torque_ratio=.006,inertia=(1e-5,2e-5,3e-5),
        cable_solref=(.01,1.),cable_solimp=(.95,.99,.001),formation_yaw=.4)
    operation=SimpleNamespace(rearrange_seconds=2.,settle_seconds=.5,unload_seconds=1.,
        hold_seconds=1.,retension_seconds=2.,restore_seconds=2.,verify_seconds=.3,
        contact_confirm_seconds=.2,gate_timeout_seconds=4.,slack_distance=.08,
        retension_microarc_deg=5.,contact_threshold=.001,support_threshold=.02,
        gap_tolerance=.003,closing_speed_limit=.15,payload_speed_limit=.08,payload_error_limit=.05)
    position=np.array([.2,-.1,1.2]);velocity=np.array([.01,.02,.03])
    observation=dict(payload_pos=position,payload_vel=velocity,
        uav_pos=position+np.array([[.25,0,.55],[-.125,.2165,.55],[-.125,-.2165,.55]]),
        uav_vel=np.repeat(velocity[None],3,axis=0),uav_rotation=np.repeat(np.eye(3)[None],3,axis=0),
        uav_omega=np.zeros((3,3)),motor_activations=np.ones((3,4))*.1,
        cable_gaps=np.ones(3)*.00003,cable_tensions=np.ones(3)*.1,
        external_forces=np.zeros((4,3)))
    memory=dict(payload_integral=np.array([.1,.2,.3]),position_integral=np.arange(9).reshape(3,3)*.01)
    return observation,plant,operation,memory


def empty_pack(width=4):
    n=len(FEATURE_NAMES)
    metadata=dict(format_version=1,width=width,feature_names=list(FEATURE_NAMES),target_names=list(TARGET_NAMES))
    arrays=dict(x_mean=np.zeros(n),x_scale=np.ones(n),y_mean=np.ones(7),y_scale=np.ones(7),
        w0=np.zeros((width,n)),b0=np.zeros(width),w1=np.zeros((width,width)),b1=np.zeros(width),
        w2=np.zeros((8,width)),b2=np.zeros(8))
    return dict(metadata=metadata,arrays=arrays)


def test_features_have_declared_layout_and_ignore_mission_goal_and_horizontal_origin():
    obs,plant,operation,memory=fixture()
    original=features(obs,plant,2,operation,True,memory)
    assert original.shape==(len(FEATURE_NAMES),) and len(set(FEATURE_NAMES))==len(FEATURE_NAMES)
    changed=copy.deepcopy(obs)
    changed['mission_goal']=np.array([999.,100.,200.])
    shift=np.array([20.,-30.,0.])
    changed['payload_pos']+=shift;changed['uav_pos']+=shift
    np.testing.assert_allclose(original,features(changed,plant,2,operation,True,memory),atol=1e-14)
    for i in range(3):assert original[FEATURE_NAMES.index(f'donor_onehot_{i}')]==float(i==2)


def test_features_retain_motor_filter_controller_memory_and_recovery_parameters():
    obs,plant,operation,memory=fixture();a=features(obs,plant,0,operation,False,memory)
    obs['motor_activations'][1,2]+=.02;memory['position_integral'][2,0]+=.1;operation.retension_seconds=3.
    b=features(obs,plant,0,operation,False,memory)
    changed=set(np.where(a!=b)[0])
    assert changed=={FEATURE_NAMES.index(x) for x in ('motor_activation_1_2',
        'controller_position_integral_2_x','operation_retension_seconds')}
    assert a[FEATURE_NAMES.index('controller_memory_observed')]==1
    none=features(obs,plant,0,operation,False)
    assert none[FEATURE_NAMES.index('controller_memory_observed')]==0


def test_nonfinite_or_wrong_shape_features_fail_before_inference():
    obs,plant,operation,memory=fixture();obs['uav_vel'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):features(obs,plant,0,operation,False,memory)
    obs,plant,operation,memory=fixture();obs['motor_activations']=np.zeros(12)
    with pytest.raises(ValueError,match='shape'):features(obs,plant,0,operation,False,memory)


def test_numpy_predictor_is_frozen_and_enforces_physical_output_ranges():
    pack=empty_pack();pack['arrays']['b2'][0]=-10.;pack['arrays']['b2'][7]=2.
    predictor=FrozenPredictor(pack)
    x=np.zeros((3,len(FEATURE_NAMES)))
    pred=predictor.predict(x)
    assert pred.shape==(3,8) and np.all(pred[:,:7]>=0) and np.all((pred[:,7]>0)&(pred[:,7]<1))
    pack['arrays']['b2'][:]=999.
    np.testing.assert_array_equal(pred,predictor.predict(x))
    with pytest.raises(ValueError):predictor.arrays['b2'][0]=5.
    np.testing.assert_array_equal(pred[0],predictor.predict(x[0]))


def test_serialization_rejects_tampered_weights_and_preserves_numpy_predictions(tmp_path):
    pack=empty_pack();prefix=tmp_path/'model'
    save_model(pack,prefix);a=FrozenPredictor(pack);b=load_model(prefix)
    x=np.ones(len(FEATURE_NAMES))
    np.testing.assert_array_equal(a.predict(x),b.predict(x))
    with pytest.raises(FileExistsError):save_model(pack,prefix)
    weight=prefix.with_suffix('.npz');weight.write_bytes(weight.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):load_model(prefix)


def synthetic_dataset(n=48):
    rng=np.random.default_rng(17)
    x=np.zeros((n,len(FEATURE_NAMES)),dtype=np.float32);x[:,:3]=rng.normal(size=(n,3))
    y=np.column_stack([np.maximum(.2+.03*x[:,j%3]+.01*j,0.) for j in range(7)]+
                      [(x[:,0]>0).astype(float)]).astype(np.float32)
    return x,y


def test_context_groups_cannot_cross_train_validation_test_boundaries(tmp_path):
    x,y=synthetic_dataset();path=tmp_path/'dataset.npz'
    np.savez(path,X_train=x[:24],Y_train=y[:24],X_val=x[24:36],Y_val=y[24:36],
        X_test=x[36:],Y_test=y[36:],groups_train=np.arange(24),groups_val=np.arange(24,36),
        groups_test=np.arange(36,48))
    assert len(load_dataset(path)['X_train'])==24
    np.savez(path,X_train=x[:24],Y_train=y[:24],X_val=x[24:36],Y_val=y[24:36],
        groups_train=np.arange(24),groups_val=np.arange(12))
    with pytest.raises(ValueError,match='overlap'):load_dataset(path)


def test_report_records_completion_class_support_even_when_all_positive():
    y=np.ones((5,8));prediction=y.copy();prediction[:,7]=.9
    metrics=prediction_metrics(prediction,y)
    assert metrics['completion_positive_labels']==5 and metrics['completion_negative_labels']==0
    assert metrics['completion_accuracy']==1 and metrics['completion_brier_score']==pytest.approx(.01)


def test_training_uses_train_only_normalization_and_numpy_matches_selected_torch_checkpoint():
    pytest.importorskip('torch')
    x,y=synthetic_dataset();validation=x[32:].copy();validation[:,0]+=4.
    pack=fit_model(x[:32],y[:32],validation,y[32:],width=8,seed=2,steps=40,validation_every=10,batch_size=16)
    np.testing.assert_allclose(pack['arrays']['x_mean'],x[:32].astype(np.float64).mean(0).astype(np.float32),rtol=0,atol=0)
    assert pack['metadata']['numpy_torch_max_absolute_difference']<2e-5
    logged=pack['metadata']['history']
    assert pack['metadata']['best_validation_loss']==min(row['validation_loss'] for row in logged)
    assert pack['metadata']['normalization']=='training split only'
    assert pack['metadata']['train_cases']==32 and pack['metadata']['validation_cases']==16
    assert np.isfinite(FrozenPredictor(pack).predict(validation)).all()


def test_training_repeats_exactly_for_a_fixed_seed_without_mutating_input_data():
    pytest.importorskip('torch')
    x,y=synthetic_dataset();before_x=x.copy();before_y=y.copy()
    first=fit_model(x[:32],y[:32],x[32:],y[32:],width=8,seed=3,steps=20,validation_every=10)
    second=fit_model(x[:32],y[:32],x[32:],y[32:],width=8,seed=3,steps=20,validation_every=10)
    for key in first['arrays']:np.testing.assert_array_equal(first['arrays'][key],second['arrays'][key])
    np.testing.assert_array_equal(x,before_x);np.testing.assert_array_equal(y,before_y)


def test_heldout_action_programs_can_share_test_contexts_but_never_training_contexts(tmp_path):
    x,y=synthetic_dataset();path=tmp_path/'dataset.npz'
    data=dict(X_train=x[:24],Y_train=y[:24],X_val=x[24:36],Y_val=y[24:36],
        X_test=x[36:],Y_test=y[36:],X_test_action=x[36:],Y_test_action=y[36:],
        groups_train=np.arange(24),groups_val=np.arange(24,36),groups_test=np.arange(36,48),
        groups_test_action=np.arange(36,48))
    np.savez(path,**data)
    assert len(load_dataset(path)['X_test_action'])==12
    data['groups_test_action']=np.arange(12)
    np.savez(path,**data)
    with pytest.raises(ValueError,match='overlap'):load_dataset(path)


def test_dotted_model_prefixes_remain_distinct(tmp_path):
    first=save_model(empty_pack(),tmp_path/'model.small.seed0')
    second=save_model(empty_pack(),tmp_path/'model.large.seed0')
    assert first['weights']!=second['weights']
    assert first['weights'].endswith('model.small.seed0.npz')
    np.testing.assert_array_equal(load_model(tmp_path/'model.small.seed0').predict(np.zeros(len(FEATURE_NAMES))),
                                  load_model(tmp_path/'model.large.seed0').predict(np.zeros(len(FEATURE_NAMES))))

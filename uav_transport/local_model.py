"""Goal-free local operation outcome models with portable NumPy inference.

Feature normalization uses training data only. Validation selects a checkpoint;
mission trajectories and targets never enter this module's feature interface.
PyTorch is imported only by training, so execution needs NumPy alone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


TARGET_NAMES = (
    "peak_payload_excursion", "peak_donor_tension", "elapsed_seconds",
    "absolute_actuator_work", "total_squared_thrust_integral",
    "final_payload_error", "final_payload_speed", "completion",
)
TARGET_UNITS = ("m", "N", "s", "J", "N^2 s", "m", "m/s", "indicator")
PLANT_SCALARS = ("dt", "control_dt", "drone_mass", "payload_mass", "rotor_max",
    "motor_tau", "cable_length", "formation_radius", "gravity", "arm_length",
    "yaw_torque_ratio")
OPERATION_SCALARS = ("rearrange_seconds", "settle_seconds", "unload_seconds",
    "hold_seconds", "retension_seconds", "restore_seconds", "verify_seconds",
    "contact_confirm_seconds", "gate_timeout_seconds", "slack_distance",
    "retension_microarc_deg", "contact_threshold", "support_threshold",
    "gap_tolerance", "closing_speed_limit", "payload_speed_limit", "payload_error_limit")


def _feature_names():
    names = ["payload_height"]+[f"payload_velocity_{axis}" for axis in "xyz"]
    for block, size in (("uav_relative_position", 3), ("uav_relative_velocity", 3),
        ("uav_body_to_world_rotation", 9), ("uav_body_angular_velocity", 3),
        ("motor_activation", 4)):
        names += [f"{block}_{i}_{j}" for i in range(3) for j in range(size)]
    names += [f"cable_gap_{i}" for i in range(3)]
    names += [f"cable_tension_{i}" for i in range(3)]
    names += [f"external_force_{i}_{axis}" for i in range(4) for axis in "xyz"]
    names += [f"donor_onehot_{i}" for i in range(3)]
    names += ["calibrated_"+x for x in PLANT_SCALARS]
    names += [f"calibrated_inertia_{axis}" for axis in "xyz"]
    names += [f"calibrated_cable_solref_{i}" for i in range(2)]
    names += [f"calibrated_cable_solimp_{i}" for i in range(3)]
    names += ["formation_yaw_sin", "formation_yaw_cos"]
    names += ["operation_"+x for x in OPERATION_SCALARS]
    names += ["operation_reconfigure"]
    names += [f"controller_payload_integral_{axis}" for axis in "xyz"]
    names += [f"controller_position_integral_{i}_{axis}" for i in range(3) for axis in "xyz"]
    names += ["controller_memory_observed"]
    return tuple(names)


FEATURE_NAMES = _feature_names()


def _get(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def features(observation, plant_config, donor, recovery_config, reconfigure,
             controller_memory=None):
    """Encode a fresh local operation, using physical state and declared inputs.

    The payload's horizontal position and spherical orientation are irrelevant
    to this central-attachment model. Its height preserves floor clearance.
    Local hold position is defined by the current physical payload position.
    Calibrated parameters must be the information actually available online.
    """
    if donor not in (0, 1, 2):
        raise ValueError("Expected one of three donor UAV indices")
    obs = observation
    position = _array(obs["payload_pos"], (3,), "payload position")
    velocity = _array(obs["payload_vel"], (3,), "payload velocity")
    uav_position = _array(obs["uav_pos"], (3, 3), "UAV position")
    uav_velocity = _array(obs["uav_vel"], (3, 3), "UAV velocity")
    parts = [position[2:3], velocity,
        (uav_position-position).reshape(-1), (uav_velocity-velocity).reshape(-1),
        _array(obs["uav_rotation"], (3,3,3), "UAV rotation").reshape(-1),
        _array(obs["uav_omega"], (3,3), "UAV angular velocity").reshape(-1),
        _array(obs["motor_activations"], (3,4), "motor activation").reshape(-1),
        _array(obs["cable_gaps"], (3,), "cable gaps"),
        _array(obs["cable_tensions"], (3,), "cable tensions"),
        _array(obs.get("external_forces", np.zeros((4,3))), (4,3), "external forces").reshape(-1),
        np.eye(3)[donor],
        np.asarray([_get(plant_config, key) for key in PLANT_SCALARS], dtype=float),
        _array(_get(plant_config,"inertia"), (3,), "calibrated inertia"),
        _array(_get(plant_config,"cable_solref"), (2,), "cable solref"),
        _array(_get(plant_config,"cable_solimp"), (3,), "cable solimp")]
    yaw = float(_get(plant_config,"formation_yaw"))
    parts += [np.array([np.sin(yaw),np.cos(yaw)]),
        np.asarray([_get(recovery_config,key) for key in OPERATION_SCALARS],dtype=float),
        np.array([float(bool(reconfigure))])]
    memory = {} if controller_memory is None else controller_memory
    parts += [_array(memory.get("payload_integral", np.zeros(3)), (3,), "payload integral"),
        _array(memory.get("position_integral", np.zeros((3,3))), (3,3), "position integral").reshape(-1),
        np.array([float(controller_memory is not None)])]
    result = np.concatenate(parts)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Local model features must be finite and match the declared layout")
    return result


def _digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((str(array.dtype),array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _xy(x, y, label):
    x, y = np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.float32)
    if x.ndim != 2 or not len(x) or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{label} X must be a nonempty matrix with the declared feature width")
    if y.shape != (len(x),len(TARGET_NAMES)):
        raise ValueError(f"{label} Y must contain the eight declared targets")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{label} inputs and targets must be finite")
    if (y[:,:7]<0).any() or not np.isin(y[:,7],(0.,1.)).all():
        raise ValueError("Physical outcomes must be nonnegative and completion labels binary")
    return x,y


class FrozenPredictor:
    """Two-hidden-layer MLP with copied read-only weights and NumPy inference."""
    _arrays = ("x_mean","x_scale","y_mean","y_scale","w0","b0","w1","b1","w2","b2")

    def __init__(self, pack):
        self.metadata = copy.deepcopy(pack["metadata"])
        if self.metadata.get("format_version") != 1:
            raise ValueError("Unknown local model format")
        if self.metadata.get("feature_names") != list(FEATURE_NAMES) or self.metadata.get("target_names") != list(TARGET_NAMES):
            raise ValueError("Local model feature or target layout does not match")
        self.arrays = {}
        for name in self._arrays:
            value = np.array(pack["arrays"][name],dtype=np.float32,copy=True)
            if not np.isfinite(value).all(): raise ValueError("Nonfinite frozen model array: "+name)
            value.flags.writeable = False
            self.arrays[name] = value
        a=self.arrays; width=int(self.metadata["width"])
        expected = dict(x_mean=(len(FEATURE_NAMES),),x_scale=(len(FEATURE_NAMES),),
            y_mean=(7,),y_scale=(7,),w0=(width,len(FEATURE_NAMES)),b0=(width,),
            w1=(width,width),b1=(width,),w2=(8,width),b2=(8,))
        if any(a[k].shape!=shape for k,shape in expected.items()):
            raise ValueError("Frozen MLP parameter shapes do not match the declared architecture")
        if (a["x_scale"]<=0).any() or (a["y_scale"]<=0).any():
            raise ValueError("Normalization scales must be positive")

    def predict(self, x):
        values=np.asarray(x,dtype=np.float32)
        single=values.ndim==1
        if single: values=values[None]
        if values.ndim!=2 or values.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError("Prediction input must match the finite feature layout")
        a=self.arrays
        h=(values-a["x_mean"])/a["x_scale"]
        h=np.maximum(h@a["w0"].T+a["b0"],0.)
        h=np.maximum(h@a["w1"].T+a["b1"],0.)
        h=h@a["w2"].T+a["b2"]
        physical=np.maximum(h[:,:7]*a["y_scale"]+a["y_mean"],0.)
        probability=1./(1.+np.exp(-np.clip(h[:,7:8],-60.,60.)))
        result=np.concatenate((physical,probability),axis=1)
        return result[0] if single else result

    __call__ = predict


def prediction_metrics(predicted, target):
    predicted, target=np.asarray(predicted),np.asarray(target)
    if predicted.shape!=target.shape or target.ndim!=2 or target.shape[1]!=8:
        raise ValueError("Expected matching eight-outcome prediction matrices")
    difference=predicted[:,:7]-target[:,:7]
    return dict(cases=len(target),rmse={name:float(value) for name,value in
        zip(TARGET_NAMES[:7],np.sqrt(np.mean(difference*difference,axis=0)))},
        mean_absolute_error={name:float(value) for name,value in
        zip(TARGET_NAMES[:7],np.mean(np.abs(difference),axis=0))},
        completion_accuracy=float(np.mean((predicted[:,7]>=.5)==target[:,7])),
        completion_brier_score=float(np.mean((predicted[:,7]-target[:,7])**2)),
        completion_positive_labels=int(target[:,7].sum()),
        completion_negative_labels=int(len(target)-target[:,7].sum()))


def fit_model(Xtrain,Ytrain,Xval,Yval,*,width=32,seed=0,steps=1500,
              batch_size=128,learning_rate=.002,validation_every=50):
    """Fit on local training outcomes and choose the best validation checkpoint."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    train_x,train_y=_xy(Xtrain,Ytrain,"training")
    val_x,val_y=_xy(Xval,Yval,"validation")
    if min(width,steps,batch_size,validation_every)<1 or learning_rate<=0:
        raise ValueError("Positive architecture and optimization parameters required")
    x_mean=train_x.astype(np.float64).mean(0).astype(np.float32);x_std=train_x.astype(np.float64).std(0)
    x_scale=np.where(x_std>1e-8,x_std,1.).astype(np.float32)
    y_mean=train_y[:,:7].astype(np.float64).mean(0).astype(np.float32);y_std=train_y[:,:7].astype(np.float64).std(0)
    y_scale=np.where(y_std>1e-8,y_std,1.).astype(np.float32)
    train_input=torch.from_numpy((train_x-x_mean)/x_scale)
    val_input=torch.from_numpy((val_x-x_mean)/x_scale)
    train_target=torch.from_numpy(np.column_stack(((train_y[:,:7]-y_mean)/y_scale,train_y[:,7])))
    val_target=torch.from_numpy(np.column_stack(((val_y[:,:7]-y_mean)/y_scale,val_y[:,7])))
    started=time.perf_counter();previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            network=nn.Sequential(nn.Linear(len(FEATURE_NAMES),width),nn.ReLU(),
                nn.Linear(width,width),nn.ReLU(),nn.Linear(width,8))
            optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)
            rng=np.random.default_rng(seed)
            best_loss=float("inf");best=None;best_step=-1;history=[]
            def objective(output, target):
                return F.mse_loss(output[:,:7],target[:,:7])+F.binary_cross_entropy_with_logits(output[:,7],target[:,7])
            def evaluate(step,train_loss):
                nonlocal best,best_loss,best_step
                network.eval()
                with torch.no_grad():loss=float(objective(network(val_input),val_target))
                history.append(dict(step=step,train_batch_loss=train_loss,validation_loss=loss))
                if loss<best_loss:
                    best_loss=loss;best_step=step
                    best={key:value.detach().clone() for key,value in network.state_dict().items()}
                network.train()
            evaluate(0,None)
            for step in range(1,steps+1):
                indices=torch.from_numpy(rng.integers(0,len(train_x),size=min(batch_size,len(train_x))))
                optimizer.zero_grad(set_to_none=True)
                loss=objective(network(train_input[indices]),train_target[indices])
                if not torch.isfinite(loss):raise FloatingPointError("Local model optimization became nonfinite")
                loss.backward();optimizer.step()
                if step%validation_every==0 or step==steps:evaluate(step,float(loss.detach()))
            network.load_state_dict(best);network.eval()
            arrays=dict(x_mean=x_mean,x_scale=x_scale,y_mean=y_mean,y_scale=y_scale)
            for layer,index in enumerate((0,2,4)):
                arrays[f"w{layer}"]=network[index].weight.detach().numpy().copy()
                arrays[f"b{layer}"]=network[index].bias.detach().numpy().copy()
            metadata=dict(format_version=1,feature_names=list(FEATURE_NAMES),target_names=list(TARGET_NAMES),
                target_units=list(TARGET_UNITS),width=int(width),seed=int(seed),steps=int(steps),
                batch_size=int(batch_size),learning_rate=float(learning_rate),best_step=best_step,
                best_validation_loss=best_loss,validation_every=int(validation_every),
                parameters=sum(p.numel() for p in network.parameters()),
                train_cases=len(train_x),validation_cases=len(val_x),normalization="training split only",
                regression="seven standardized physical outcomes; negative predictions clipped to zero at inference",
                completion="sigmoid completion score trained with binary cross entropy",
                controller_conditioning="observed integral memory; controller gains are fixed by the dataset registration",
                checkpoint_selection="minimum validation standardized-MSE plus completion-BCE; no test or mission labels",
                constant_training_features=[FEATURE_NAMES[i] for i in np.where(x_std<=1e-8)[0]],
                data_sha256=dict(Xtrain=_digest(train_x),Ytrain=_digest(train_y),Xval=_digest(val_x),Yval=_digest(val_y)),
                history=history,training_seconds=time.perf_counter()-started)
            pack=dict(metadata=metadata,arrays=arrays)
            predictor=FrozenPredictor(pack)
            predicted=predictor.predict(val_x)
            with torch.no_grad():
                output=network(val_input)
                torch_output=torch.cat((torch.clamp(output[:,:7]*torch.from_numpy(y_scale)+torch.from_numpy(y_mean),min=0.),
                                        torch.sigmoid(output[:,7:8])),dim=1).numpy()
            np.testing.assert_allclose(predicted,torch_output,rtol=2e-5,atol=2e-5)
            metadata["numpy_torch_max_absolute_difference"]=float(np.max(np.abs(predicted-torch_output)))
            metadata["training_metrics"]=prediction_metrics(predictor.predict(train_x),train_y)
            metadata["validation_metrics"]=prediction_metrics(predicted,val_y)
            return pack
    finally:
        torch.set_num_threads(previous_threads)


def _model_paths(prefix):
    prefix=Path(prefix)
    if prefix.suffix in (".json",".npz"):prefix=prefix.with_suffix("")
    return Path(str(prefix)+".npz"),Path(str(prefix)+".json")


def save_model(pack,prefix):
    predictor=FrozenPredictor(pack)
    array_path,metadata_path=_model_paths(prefix)
    if array_path.exists() or metadata_path.exists():
        raise FileExistsError("Preserve existing frozen models with a fresh output prefix")
    array_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(array_path,**predictor.arrays)
    metadata=copy.deepcopy(pack["metadata"])
    metadata.update(array_file=array_path.name,array_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    return dict(weights=str(array_path),metadata=str(metadata_path),sha256=metadata["array_sha256"])


def load_model(prefix):
    array_path,metadata_path=_model_paths(prefix)
    metadata=json.loads(metadata_path.read_text())
    if metadata["array_file"]!=array_path.name or hashlib.sha256(array_path.read_bytes()).hexdigest()!=metadata["array_sha256"]:
        raise ValueError("Frozen local model artifact does not match its metadata hash")
    with np.load(array_path,allow_pickle=False) as loaded:
        arrays={key:loaded[key] for key in FrozenPredictor._arrays}
    return FrozenPredictor(dict(metadata=metadata,arrays=arrays))


def load_dataset(path):
    """Read explicit splits, validating context separation when IDs are supplied."""
    with np.load(path,allow_pickle=False) as source:
        dataset={key:source[key] for key in source.files}
    for split in ("train","val"):
        _xy(dataset[f"X_{split}"],dataset[f"Y_{split}"],split)
    splits=[split for split in ("train","val","test","test_action") if f"X_{split}" in dataset]
    groups={}
    for split in splits:
        _xy(dataset[f"X_{split}"],dataset[f"Y_{split}"],split)
        key="groups_"+split
        if key in dataset:
            values=np.asarray(dataset[key])
            if values.shape!=(len(dataset[f"X_{split}"]),):raise ValueError("Context IDs must align with local trajectories")
            groups[split]=set(values.astype(str).tolist())
    if groups and len(groups)!=len(splits):
        raise ValueError("Context IDs must cover every supplied data split")
    for i,split in enumerate(splits):
        for other in splits[i+1:]:
            if groups and not (split.startswith("test") and other.startswith("test")) and groups[split]&groups[other]:
                raise ValueError("Local context IDs overlap between "+split+" and "+other)
    return dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--width",type=int,default=32)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--steps",type=int,default=1500)
    parser.add_argument("--batch-size",type=int,default=128)
    args=parser.parse_args()
    data=load_dataset(args.data)
    pack=fit_model(data['X_train'],data['Y_train'],data['X_val'],data['Y_val'],
        width=args.width,seed=args.seed,steps=args.steps,batch_size=args.batch_size)
    pack['metadata']['dataset_path']=str(args.data)
    pack['metadata']['dataset_sha256']=hashlib.sha256(args.data.read_bytes()).hexdigest()
    for split in ('test','test_action'):
        if 'X_'+split in data:
            pack['metadata'][split+'_metrics']=prediction_metrics(FrozenPredictor(pack).predict(data['X_'+split]),data['Y_'+split])
    paths=save_model(pack,args.out)
    print(json.dumps(dict(**paths,best_step=pack['metadata']['best_step'],
        validation=pack['metadata']['validation_metrics'],test=pack['metadata'].get('test_metrics'),
        test_action=pack['metadata'].get('test_action_metrics'))),flush=True)


if __name__=='__main__':
    main()

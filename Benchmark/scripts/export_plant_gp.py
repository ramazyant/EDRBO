"""Convert the plant leaf-area GPs of Tay et al. (gpflow SVGP pickles) to numpy files used by
Benchmark.Real_World.Plant_Leaf (no TensorFlow needed at run time).

Sources (MIT licence, (c) sebtsh):
  https://github.com/sebtsh/efficient-distributionally-robust-BO  data/plant/NH3pH_gp_leaf_dict.p (+ _req_variables.p)
  https://github.com/sebtsh/unified-framework-BOCU               data/plant/plant_funcs.py (model constructor)
The pickles reference `tensorflow.python.training.tracking`, renamed in later TensorFlow versions; the
unpickler below maps it to `tensorflow.python.trackable`. Only the predictive mean of the location
latent of the heteroscedastic model is needed: mean(x) = K(x, Z) L^{-T} q_mu (whitened SVGP),
stored as alpha = L^{-T} q_mu. The script checks the numpy prediction against gpflow's predict_y.

    pip install gpflow tensorflow tensorflow-probability tf-keras
    python Benchmark/scripts/export_plant_gp.py --edrbo /path/efficient-distributionally-robust-BO --bocu /path/unified-framework-BOCU
"""
import argparse
import importlib.util
import os
import pickle

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('tensorflow.python.training.tracking'):
            module = module.replace('tensorflow.python.training.tracking', 'tensorflow.python.trackable')
        return super().find_class(module, name)


def kse(A, B, var, ls):
    A, B = A / ls, B / ls
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return var * np.exp(-0.5 * np.maximum(d2, 0))


def main():
    import gpflow
    import tensorflow as tf
    import tensorflow_probability as tfp
    ap = argparse.ArgumentParser()
    ap.add_argument('--edrbo', required=True)
    ap.add_argument('--bocu', required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('pf', os.path.join(a.bocu, 'data', 'plant', 'plant_funcs.py'))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    softplus = lambda z: np.logaddexp(0, z)
    jobs = [(os.path.join(a.edrbo, 'data', 'plant', 'NH3pH_gp_leaf_dict.p'), 174, 2, 'plant_leaf_NH3pH_svgp.npz',
             67.2466342112483, 59.347376136036964, ['NH3_uM', 'pH'])]
    for fn, M, d, out, lm, lsd, names in jobs:
        raw = _Unpickler(open(fn, 'rb')).load()
        P = {k: np.asarray(v.pretransformed_input.numpy(), dtype=np.float64) for k, v in raw.items()}
        var0, ls0 = softplus(P['.kernel.kernels[0].variance']), softplus(P['.kernel.kernels[0].lengthscales'])
        var1, ls1 = softplus(P['.kernel.kernels[1].variance']), softplus(P['.kernel.kernels[1].lengthscales'])
        Z0 = P['.inducing_variable.inducing_variable_list[0].Z']
        Z1 = P['.inducing_variable.inducing_variable_list[1].Z']
        q_mu = P['.q_mu']
        q_sqrt = tfp.math.fill_triangular(tf.constant(P['.q_sqrt'])).numpy()
        L = np.linalg.cholesky(kse(Z0, Z0, var0, ls0) + gpflow.config.default_jitter() * np.eye(M))
        alpha = np.linalg.solve(L.T, q_mu[:, 0])
        np.savez_compressed(os.path.join(OUT, out), Z=Z0, alpha=alpha, variance=var0, lengthscales=ls0,
                            leaf_mean=lm, leaf_std=lsd, names=np.array(names))
        gp = pf.init_heteroscedastic_gp(num_inducing=M, d=d)
        gpflow.utilities.multiple_assign(gp, {
            '.kernel.kernels[0].variance': var0, '.kernel.kernels[0].lengthscales': ls0,
            '.kernel.kernels[1].variance': var1, '.kernel.kernels[1].lengthscales': ls1,
            '.inducing_variable.inducing_variable_list[0].Z': Z0, '.inducing_variable.inducing_variable_list[1].Z': Z1,
            '.q_mu': q_mu, '.q_sqrt': q_sqrt})
        X = Z0.min(0) + (Z0.max(0) - Z0.min(0)) * np.random.default_rng(0).random((500, d))
        err = np.abs(kse(X, Z0, var0, ls0) @ alpha - gp.predict_y(X)[0].numpy()[:, 0]).max()
        print(f'{out}: max |numpy - gpflow| = {err:.2e} (standardised units)')


if __name__ == '__main__':
    main()


import numpy as np
import csv
import torch
import time
import sys

model_type = 'mlp'

def run(f, optimizer_theta, maxite=1e5, dispspan=1, logspan=1, log_file=None):
    t = 1
    gloss = np.inf
    train_start = time.time()

    if log_file is not None:
        with open(log_file, 'w') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow(['iteratoin', 'eval', 'gloss'] + optimizer_theta.log_header(theta_log=False))

    while t <= maxite:
        iter_start = time.time()
        lam = optimizer_theta.get_lam()
        losses = np.array([])
        general_losses = np.array([])

        sample_c = np.array([optimizer_theta.sampling() for _ in range(lam)], dtype=np.int32)
        for i in range(lam):
            trace = [str(i) for i in np.where(sample_c[i])[1]]
            loss = f.eval_mlp(trace, eval_trace = True)
            losses = np.append(losses, loss.detach().cpu().numpy())
            general_losses = np.append(general_losses, loss.detach().cpu().numpy())
        mean_loss = loss.mean()
        f.update_mlp(mean_loss)

        sample_c = np.array([optimizer_theta.sampling() for _ in range(lam)], dtype=np.int32)
        for i in range(lam):
            trace = [str(i) for i in np.where(sample_c[i])[1]]
            loss = f.eval_mlp(trace, eval_trace = True)
            losses = np.append(losses, loss.detach().cpu().numpy())
            general_losses = np.append(general_losses, loss.detach().cpu().numpy())
        optimizer_theta.update(sample_c, losses[-lam:])
        iter_end = time.time()

        # logging
        gloss = np.min(general_losses)
        if t % dispspan == 0 or t == 1 or t + 1 == maxite:
            print('time: {:.2f} ite: {} gloss {:f} iter_time: {:.2f} theta_convergence: {:.4f} theta_mle: {:d} delta: {}'.format(
                iter_end - train_start, t, gloss, iter_end - iter_start,optimizer_theta.theta.max(axis=1).mean(), int(optimizer_theta.mle()[:, 0].sum()),
                optimizer_theta.get_delta()))
            sys.stdout.flush()
        if log_file is not None and (t % logspan == 0 or t == 1):
            with open(log_file, 'a') as fp:
                writer = csv.writer(fp, lineterminator='\n')
                writer.writerow([t, gloss] + optimizer_theta.log(theta_log=False))
        if t % 5 == 0:
            eval_trace = np.array([optimizer_theta.sampling() for _ in range(50)], dtype=np.int32)
            np.save(f'trace/{model_type}_{t}.npy', eval_trace)
        t += 1

    print('ite: {} gloss {:f} theta_convergence: {:.4f} theta_mle: {:d} delta: {}'.format(
        t, gloss, optimizer_theta.theta.max(axis=1).mean(), int(optimizer_theta.mle()[:, 0].sum()),
        optimizer_theta.get_delta()))

    if log_file is not None:
        with open(log_file, 'a') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow([t, gloss] + optimizer_theta.log(theta_log=False))

    return

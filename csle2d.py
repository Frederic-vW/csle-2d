#!/usr/bin/python3
# -*- coding: utf-8 -*-
# last tested Python version: 3.6.9
# Complex Stuart-Landau Equation model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import cv2

def csle2d(N, T, t0, dt, s, D, mu0, mu1):
    # initialize CSLE system
    z = np.zeros((N,N),dtype=np.complex)
    z2 = np.zeros((N,N),dtype=np.complex)
    dz = np.zeros((N,N),dtype=np.complex)
    s_sqrt_dt = s*np.sqrt(dt)
    Z = np.zeros((T,N,N),dtype=np.complex)
    # bifurcation parameter time course
    u = mu0*np.ones(t0)
    v = np.linspace(mu0,mu1,num=T)
    mu = np.hstack((u,v))
    # iterate
    for t in range(t0+T):
        if (t%100 == 0): print("    t = ", t, "\r", end="")
        # CSLE equations
        z2 = z * np.conj(z)
        dz = (mu[t] + 1j)*z - z*z2 + D*L(z)
        # stochastic integration
        n_re = np.random.randn(N,N)
        n_im = np.random.randn(N,N)
        z += (dz*dt + s_sqrt_dt*(n_re + 1j*n_im))
        if (t >= t0):
            Z[t-t0,:,:] = z
    print("\n")
    return Z


def animate_pyplot1(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.FuncAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/simple_anim.html
    (Faster than animate_pyplot2)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    vmin, vmax = data.min(), data.max()
    # setup animation image
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    t = plt.imshow(data[0,:,:], origin="lower", cmap=plt.cm.gray, \
                   vmin=vmin, vmax=vmax)
    plt.tight_layout()
    # frame generator
    print("[+] Animate")
    def animate(i):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        t.set_data(data[i,:,:])
    # create animation
    ani = animation.FuncAnimation(fig, animate, frames=n1, interval=10)
    #ani.save(fname)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_pyplot2(fname, data, downsample=10):
    """
    Animate 3D array as .mp4 using matplotlib.animation.ArtistAnimation
    modified from:
    https://matplotlib.org/stable/gallery/animation/dynamic_image.html
    (Slower than animate_pyplot1)

    Args:
        fname : string
            save result in mp4 format as `fname` in the current directory
        data : array (nt,nx,ny) (time, space, space)
        downsample : integer, optional
            temporal downsampling factor

    """
    nt, nx, ny = data.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, data, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        data = f_ip(t1)
    print("[+] Animate")
    vmin, vmax = data.min(), data.max()
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.axis("off")
    plt.tight_layout()
    ims = []
    for i in range(n1):
        if (i%10 == 0): print(f"    i = {i:d}/{n1:d}\r", end="")
        im = ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                       vmin=vmin, vmax=vmax, animated=True)
        if i == 0:
            ax.imshow(data[i,:,:], origin="lower", cmap=plt.cm.gray, \
                           vmin=vmin, vmax=vmax)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    #ani.save(fname2)
    writer = animation.FFMpegWriter(fps=15, bitrate=1200)
    ani.save(fname, writer=writer)
    plt.show()
    print("\n")


def animate_video(fname, x, downsample=None):
    nt, nx, ny = x.shape
    n1 = nt
    if downsample:
        n1 = int(nt/downsample) # number of samples after downsampling
        t0 = np.arange(nt)
        f_ip = interp1d(t0, x, axis=0, kind='linear')
        t1 = np.linspace(0, nt-1, n1)
        x = f_ip(t1)
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    #print(f"n1 = {n1:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    print("[+] Animate")
    frate = 30
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    for i in range(n1):
        print(f"    i = {i:d}/{n1:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("Complex Stuart-Landau Equation (CSLE) lattice model\n")
    N = 128
    T = 2500
    t0 = 0
    dt = 0.05
    s = 0.05
    D = 1.0
    mu0 = -0.05
    mu1 = 0.5
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", T)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise intensity s: ", s)
    print("[+] Diffusion coefficient D: ", D)
    print("[+] CSLE parameter mu0: ", mu0, ", mu1: ", mu1)

    # run simulation
    Z = csle2d(N, T, t0, dt, s, D, mu0, mu1)
    #data = real.(Z)
    data = np.angle(Z)
    print("[+] Data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(np.abs(Z), (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m)
    plt.show()

    # save data
    fname1 = f"csle2d_mu0_{mu0:.2f}_mu1_{mu1:.2f}_s_{s:.2f}_D_{D:.2f}.npy"
    #np.save(fname1, data)
    #print("[+] Data saved as: ", fname1)

    #video
    fname2 = f"csle2d_mu0_{mu0:.2f}_mu1_{mu1:.2f}_s_{s:.2f}_D_{D:.2f}.mp4"
    #animate_pyplot1(fname2, data, downsample=4)
    #animate_pyplot2(fname2, data, downsample=4)
    animate_video(fname2, data, downsample=4) # fastest
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()

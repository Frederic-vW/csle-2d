#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Complex Stuart-Landau Equation model on a 2D lattice
# FvW 03/2018

using Interpolations
using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function csle2d(N, T, t0, dt, s, D, mu0, mu1)
    # initialize CSLE system
    z = zeros(ComplexF64,N,N)
    z2 = zeros(ComplexF64,N,N)
    dz = zeros(ComplexF64,N,N)
    s_sqrt_dt = s*sqrt(dt)
    Z = zeros(ComplexF64,T,N,N)
	# bifurcation parameter time course
    u = mu0.*ones(t0)
    v = [_ for _ in range(mu0,stop=mu1,length=T)]
    mu = vcat(u,v) # 1-dim time course of mu
    # iterate
    for t in range(1, stop=t0+T)
        (t%100 == 0) && print("    t = ", t, "\r")
        # CSLE equations
        z2 = z .* conj(z)
        dz = (mu[t] .+ 1im).*z - z.*z2 + D.*L(z)
        # stochastic integration
        z += (dz*dt + s_sqrt_dt .* randn(ComplexF64,N,N))
        (t > t0) && (Z[t-t0,:,:] = z)
    end
    println("\n")
    return Z
end

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    println("[+] Animate")
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
               vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data, downsample=10)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    n1 = nt
    if (downsample > 0)
        println("[+] Downsampling ", downsample)
        n1 = Int(nt/downsample) # number of samples after downsampling
        data1 = zeros(Float64,n1,nx,ny)
        t0 = range(1,stop=nt)
        t1 = range(1,stop=nt,length=n1)
        for i in range(1,stop=nx)
            for j in range(1,stop=ny)
                y = data[:,i,j]
                f_ip = LinearInterpolation(t0, y)
                data1[:,i,j] = f_ip(t1)
            end
        end
    else
        data1 = copy(data)
    end
    # BW
    y = UInt8.(round.(255*(data .- minimum(data)) /
                          (maximum(data)-minimum(data))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data .- minimum(data)) /
    #                             (maximum(data)-minimum(data))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=30
    T = size(data,1)
    println("[+] Animate")
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=T,step=1)
            (i%100 == 0) && print("    i = ", i, "/", T, "\r")
            write(writer, y[i,end:-1:1,:])
        end
    end
    println("\n")
end

function L(x)
    # Laplace operator
    # periodic boundary conditions
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
    # non-periodic boundary conditions
    Lx[1,:] .= 0.0
    Lx[end,:] .= 0.0
    Lx[:,1] .= 0.0
    Lx[:,end] .= 0.0
    return Lx
end

function main()
    println("Complex Stuart-Landau Equation (CSLE) lattice model\n")
    N = 128
    T = 2500
    t0 = 0
    dt = 0.05
    s = 0.05
    D = 1.0
    mu0 = -0.05
    mu1 = 0.5
    println("[+] Lattice size N: ", N)
    println("[+] Time steps T: ", T)
    println("[+] Warm-up steps t0: ", t0)
    println("[+] Integration time step dt: ", dt)
    println("[+] Noise intensity s: ", s)
    println("[+] Diffusion coefficient D: ", D)
    println("[+] CSLE parameter mu0: ", mu0, ", mu1: ", mu1)

    # run simulation
    Z = csle2d(N, T, t0, dt, s, D, mu0, mu1)
    data = angle.(Z)
    println("[+] Data dimensions: ", size(Z))

	# plot mean voltage
    m = mean(reshape(abs.(Z), (T,N*N)), dims=2)
    plot(m, "-k"); show()

	# save data
    mu0_str = rpad(mu0, 4, '0') # bif. parameter as 4-char string
    mu1_str = rpad(mu1, 4, '0') # bif. parameter as 4-char string
    s_str = rpad(s, 4, '0') # noise as 4-char string
    D_str = rpad(D, 4, '0') # diffusion coefficient as 4-char string
    fname1 = string("csle2d_mu0_", mu0_str, "_mu1_", mu1_str, "_s_", s_str,
					"_D_", D_str, ".npy")
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", fname1)

	fname2 = string("csle2d_mu0_", mu0_str, "_mu1_", mu1_str, "_s_", s_str,
					"_D_", D_str, ".mp4")
    animate_video(fname2, data, 4)
    println("[+] Data saved as: ", fname2)
end

main()

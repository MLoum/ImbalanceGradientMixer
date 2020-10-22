# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:56:10 2020

@author: utilisateur


Systematic modeling of microfluidic concentration
gradient generators
To cite this article: Yi Wang et al 2006 J. Micromech. Microeng. 16 2128
"""

import numpy as np
import matplotlib.pyplot as plt


# Rhydro
def calculate_R_hydro(L, w, h, eta):
    return 8 * eta * L / (np.pi * w ** 4)
    beta = w / h
    sum_ = 0
    for i in range(20):
        if i % 2 != 0:
            sum_ += np.tanh(i * np.pi / 2 * beta) / i ** 5
    R = 12 * beta * L * eta / (w ** 4 * (1 - 192 * beta / np.pi ** 5 * sum_))
    return R


def combiner(d_l, d_r, q_l, q_r, name=None):
    # print("combiner " + name)
    inf_sum = len(d_l)
    s = abs(q_l) / (abs(q_l) + abs(q_r))
    # print("s : ", s)
    d_out = np.zeros(d_l.size)
    for n in range(len(d_l)):
        if n == 0:
            d_out[n] = d_l[0] * s + d_r[0] * (1 - s)
        else:
            sum_1 = 0
            sum_2 = 0
            sum_3 = 0
            for m in range(inf_sum):
                if m != n * s:
                    f1 = (m - n * s) * np.pi
                    f2 = (m + n * s) * np.pi
                    sum_1 += d_l[m] * (f1 * np.sin(f2) + f2 * np.sin(f1) / (f1 * f2))
                else:
                    sum_1 += d_l[m]
            sum_1 *= s

            for m in range(inf_sum):
                if m == n * (1 - s):
                    sum_3 += np.power(-1.0, n - m) * d_r[m]
                else:
                    F1 = (m + n - n * s) * np.pi
                    F2 = (m - n + n * s) * np.pi
                    sum_2 += d_r[m] * (np.cos(F2 / 2) * np.sin(F1 / 2) / F1 + np.cos(F1 / 2) * np.sin(F2 / 2) / F2)
            sum_2 *= 2 * (1 - s) * np.power(-1.0, n)
            sum_3 *= (1 - s)
            d_out[n] = sum_1 + sum_2 + sum_3
    return d_out


def splitter(d_in, q_l, q_r, name=None):
    # print("splitter " + name)
    d_l = np.zeros(d_in.size)
    d_r = np.zeros(d_in.size)

    inf_sum = len(d_l)
    s = abs(q_l) / (abs(q_l) + abs(q_r))
    # print("s : ", s)

    for n in range(len(d_l)):
        if n == 0:
            sum_l, sum_r = 0, 0
            for m in range(1, inf_sum):
                phi_1 = m * s * np.pi
                phi_2 = m * (1 - s) * np.pi
                sum_l += d_in[m] * np.sin(phi_1) / phi_1
                sum_r += d_in[m] * np.sin(phi_1) / phi_2

            d_l[0] = d_in[0] + sum_l
            d_r[0] = d_in[0] - sum_r
        else:
            sum_1 = 0
            for m in range(inf_sum):
                if m != n / s:
                    f1 = (n - m * s) * np.pi
                    f2 = (n + m * s) * np.pi
                    phi_1 = m * s * np.pi
                    phi_2 = m * (1 - s) * np.pi
                    sum_1 += d_in[m] * np.power(-1.0, n + 1) * phi_1 * np.sin(phi_1) / (f1 * f2)
                else:
                    sum_1 += d_in[m] / 2
            sum_1 *= 2
            d_l[n] = sum_1

            sum_r = 0
            for m in range(inf_sum):
                if m != n / (1 - s):
                    F1 = (n + m - m * s) * np.pi
                    F2 = (n - m + m * s) * np.pi
                    phi_1 = m * s * np.pi
                    phi_2 = m * (1 - s) * np.pi
                    sum_r += d_in[m] * phi_2 * np.sin(phi_1) / (F1 * F2)
                else:
                    sum_r += np.power(-1.0, m - n) * d_in[m] / 2
            sum_r *= 2
            d_r[n] = sum_r

    return d_l, d_r


def diffusion(d_in, L, w, U, D):
    d_out = np.zeros(d_in.size)
    Peclet = U * w / D
    tau = L / (w * Peclet)
    for n in range(len(d_in)):
        d_out[n] = d_in[n] * np.exp(-(n * np.pi) ** 2 * tau)

    return d_out


def calculate_output_gradient(sigma, is_debug=False):
    # Initialize concentration profile
    nb_harmonic = 10
    d_A = np.zeros(nb_harmonic)
    d_B = np.zeros(nb_harmonic)
    d_A[0] = 1

    # geometry
    h = 50E-6
    w = 200E-6
    L_entry = 1000E-6
    L_inter = 10E-6
    L_mixer = 50000000E-6
    D = 4E-10
    eta_water = 1E-3

    R00 = calculate_R_hydro(L_entry, w, h, eta_water)
    R01 = calculate_R_hydro(L_entry, w, h, eta_water)

    R00 *= (1 + np.random.normal(scale=sigma))
    R01 *= (1 + np.random.normal(scale=sigma))

    if R00 < 0:
        R00 = 0
    if R01 < 0:
        R01 = 0

    Y00 = 1 / R00
    Y01 = 1 / R01

    Ri10 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri11 = calculate_R_hydro(L_inter, w, h, eta_water)

    Ri10 *= (1 + np.random.normal(scale=sigma))
    Ri11 *= (1 + np.random.normal(scale=sigma))

    if Ri10 < 0:
        Ri10 = 0
    if Ri11 < 0:
        Ri11 = 0

    Yi10 = 1 / Ri10
    Yi11 = 1 / Ri11

    # TODO + Rinter pour les latérales ?
    R10 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R11 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R12 = calculate_R_hydro(L_mixer, w, h, eta_water)

    R10 *= (1 + np.random.normal(scale=sigma))
    R11 *= (1 + np.random.normal(scale=sigma))
    R12 *= (1 + np.random.normal(scale=sigma))

    if R10 < 0:
        R10 = 0
    if R11 < 0:
        R11 = 0
    if R12 < 0:
        R12 = 0

    Y10 = 1 / R10
    Y11 = 1 / R11
    Y12 = 1 / R12

    R20 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R21 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R22 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R23 = calculate_R_hydro(L_mixer, w, h, eta_water)

    R20 *= (1 + np.random.normal(scale=sigma))
    R21 *= (1 + np.random.normal(scale=sigma))
    R22 *= (1 + np.random.normal(scale=sigma))
    R23 *= (1 + np.random.normal(scale=sigma))

    if R20 < 0:
        R20 = 0
    if R21 < 0:
        R21 = 0
    if R22 < 0:
        R22 = 0
    if R23 < 0:
        R23 = 0

    Y20 = 1 / R20
    Y21 = 1 / R21
    Y22 = 1 / R22
    Y23 = 1 / R23

    Ri20 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri21 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri22 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri23 = calculate_R_hydro(L_inter, w, h, eta_water)

    Ri20 *= (1 + np.random.normal(scale=sigma))
    Ri21 *= (1 + np.random.normal(scale=sigma))
    Ri22 *= (1 + np.random.normal(scale=sigma))
    Ri23 *= (1 + np.random.normal(scale=sigma))

    if Ri20 < 0:
        Ri20 = 0
    if Ri21 < 0:
        Ri21 = 0
    if Ri22 < 0:
        Ri22 = 0
    if Ri23 < 0:
        Ri23 = 0

    Yi20 = 1 / Ri20
    Yi21 = 1 / Ri21
    Yi22 = 1 / Ri22
    Yi23 = 1 / Ri23

    R30 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R31 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R32 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R33 = calculate_R_hydro(L_mixer, w, h, eta_water)
    R34 = calculate_R_hydro(L_mixer, w, h, eta_water)

    R30 *= (1 + np.random.normal(scale=sigma))
    R31 *= (1 + np.random.normal(scale=sigma))
    R32 *= (1 + np.random.normal(scale=sigma))
    R33 *= (1 + np.random.normal(scale=sigma))
    R34 *= (1 + np.random.normal(scale=sigma))

    if R30 < 0:
        R30 = 0
    if R31 < 0:
        R31 = 0
    if R32 < 0:
        R32 = 0
    if R33 < 0:
        R33 = 0
    if R34 < 0:
        R34 = 0

    Y30 = 1 / R30
    Y31 = 1 / R31
    Y32 = 1 / R32
    Y33 = 1 / R33
    Y34 = 1 / R34

    Ri30 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri31 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri32 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri33 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri34 = calculate_R_hydro(L_inter, w, h, eta_water)
    Ri35 = calculate_R_hydro(L_inter, w, h, eta_water)

    Ri30 *= (1 + np.random.normal(scale=sigma))
    Ri31 *= (1 + np.random.normal(scale=sigma))
    Ri32 *= (1 + np.random.normal(scale=sigma))
    Ri33 *= (1 + np.random.normal(scale=sigma))
    Ri34 *= (1 + np.random.normal(scale=sigma))
    Ri35 *= (1 + np.random.normal(scale=sigma))

    if Ri30 < 0:
        Ri30 = 0
    if Ri31 < 0:
        Ri31 = 0
    if Ri32 < 0:
        Ri32 = 0
    if Ri33 < 0:
        Ri33 = 0
    if Ri34 < 0:
        Ri34 = 0
    if Ri35 < 0:
        Ri35 = 0

    Yi30 = 1 / Ri30
    Yi31 = 1 / Ri31
    Yi32 = 1 / Ri32
    Yi33 = 1 / Ri33
    Yi34 = 1 / Ri34
    Yi35 = 1 / Ri35

    # En pascal
    VA = 10
    VB = 10
    if is_debug:
        print("R00, R01  :", R00, R01)
        print("Ri10, Ri11  :", Ri10, Ri11)
        print("R10, R11, R12  :", R10, R11, R12)
    # V10 V11 V12 V20 V21 V22 V23 V24
    A = np.array([
        [Y10 + Y00 + Yi10, -Yi10, 0, -Y10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # V10
        [-Yi10, Y11 + Yi10 + Yi11, -Yi11, 0, 0, -Y11, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # V11
        [0, -Yi11, Y12 + Y01 + Yi11, 0, 0, 0, 0, -Y12, 0, 0, 0, 0, 0, 0, 0],  # V12
        [-Y10, 0, 0, Y20 + Y10 + Yi20, -Yi20, 0, 0, 0, -Y20, 0, 0, 0, 0, 0, 0],  # V20
        [0, 0, 0, -Yi20, Yi20 + Yi21 + Y21, -Yi21, 0, 0, 0, 0, -Y21, 0, 0, 0, 0],  # V21
        [0, -Y11, 0, 0, -Yi21, Yi21 + Yi22 + Y11, -Yi22, 0, 0, 0, 0, 0, 0, 0, 0],  # V22
        [0, 0, 0, 0, 0, -Yi22, Yi22 + Yi23 + Y22, -Yi23, 0, 0, 0, 0, -Y22, 0, 0],  # V23
        [0, 0, -Y12, 0, 0, 0, -Yi23, Yi23 + Y12 + Y23, 0, 0, 0, 0, 0, 0, -Y23],  # V24
        [0, 0, 0, -Y20, 0, 0, 0, 0, Y30 + Y20 + Yi30, -Yi30, 0, 0, 0, 0, 0],  # V30
        [0, 0, 0, 0, 0, 0, 0, 0, -Yi30, Yi30 + Y31 + Yi31, -Yi31, 0, 0, 0, 0],  # V31
        [0, 0, 0, 0, -Y21, 0, 0, 0, 0, -Yi31, Yi31 + Y21 + Yi32, -Yi32, 0, 0, 0],  # V32
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Yi32, Yi32 + Y32 + Yi33, -Yi33, 0, 0],  # V33
        [0, 0, 0, 0, 0, -Y22, 0, 0, 0, 0, 0, -Yi33, Yi33 + Y22 + Yi34, -Yi34, 0],  # V34
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -Yi34, Yi34 + Y33 + Yi35, -Yi35],  # V35
        [0, 0, 0, 0, 0, 0, -Y23, 0, 0, 0, 0, 0, 0, -Yi35, Yi35 + Y23 + Y34]  # V36
    ]
    )

    B = np.array([VA * Y00, 0, VB * Y01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # print(np.shape(A))
    # print(np.shape(B))

    V10, V11, V12, V20, V21, V22, V23, V24, V30, V31, V32, V33, V34, V35, V36 = np.linalg.solve(A, B)

    # First stage
    i00 = VA / R00
    i01 = VB / R01

    d00 = diffusion(d_A, L_entry, w, U=i00 / (w * h), D=D)
    d01 = diffusion(d_B, L_entry, w, U=i01 / (w * h), D=D)

    # Second stage
    ii10 = (V11 - V10) / Ri10
    ii11 = (V12 - V11) / Ri11

    i10 = (V10 - V20) / R10
    i11 = (V11 - V22) / R11
    i12 = (V12 - V24) / R12

    d10, di10 = splitter(d_in=d00, q_l=i10, q_r=ii10, name="Second Stage - Left : d_in=d00, q_l=i10, q_r=ii10")
    di11, d12 = splitter(d_in=d01, q_l=ii11, q_r=i12, name="Second Stage - Right : d_in=d01, q_l=ii11, q_r=i12")

    # Petite diffusion dans le canal intermédiaire
    di10 = diffusion(di10, L_inter, w, U=ii10 / (w * h), D=D)
    di11 = diffusion(di11, L_inter, w, U=ii11 / (w * h), D=D)

    d11 = combiner(d_l=di10, d_r=di11, q_l=ii10, q_r=ii11,
                   name="Second Stage - Center : d_l=di10, d_r=di11, q_l=ii10, q_r=ii11")

    # diffusion dans les mixer
    d10_out = diffusion(d10, L_mixer, w, U=i10 / (w * h), D=D)
    d11_out = diffusion(d11, L_mixer, w, U=i11 / (w * h), D=D)
    d12_out = diffusion(d12, L_mixer, w, U=i12 / (w * h), D=D)

    d10, d11, d12 = d10_out, d11_out, d12_out

    # Third stage

    ii20 = (V21 - V20) / Ri20
    ii21 = (V22 - V21) / Ri21
    ii22 = (V23 - V22) / Ri22
    ii23 = (V24 - V23) / Ri23

    i20 = V20 / R20
    i21 = V21 / R21
    i22 = V23 / R22
    i23 = V24 / R23

    d20, di20 = splitter(d_in=d10, q_l=i20, q_r=ii20, name="Third Stage - Left : d_in=d10, q_l=i20, q_r=ii20")
    di21, di22 = splitter(d_in=d11, q_l=ii21, q_r=ii22, name="Third Stage - Center : d_in=d11, q_l=ii21, q_r=ii22")
    di23, d23 = splitter(d_in=d12, q_l=ii23, q_r=i23, name="Third Stage - Right : d_in=d12, q_l=ii23, q_r=i23")

    # Petite diffusion dans le canal intermédiaire
    di20 = diffusion(di20, L_inter, w, U=ii20 / (w * h), D=D)
    di21 = diffusion(di21, L_inter, w, U=ii21 / (w * h), D=D)

    d21 = combiner(d_l=di20, d_r=di21, q_l=ii20, q_r=ii21,
                   name="Third Stage - Left : d_l=di20, d_r=di21, q_l=ii20, q_r=ii21")

    # Petite diffusion dans le canal intermédiaire
    di22 = diffusion(di22, L_inter, w, U=ii22 / (w * h), D=D)
    di23 = diffusion(di23, L_inter, w, U=ii23 / (w * h), D=D)

    d22 = combiner(d_l=di22, d_r=di23, q_l=ii22, q_r=ii23,
                   name="Third Stage - Right : d_l=di22, d_r=di23, q_l=ii22, q_r=ii23")

    # diffusion dans les mixer
    d20_out = diffusion(d20, L_mixer, w, U=i20 / (w * h), D=D)
    d21_out = diffusion(d21, L_mixer, w, U=i21 / (w * h), D=D)
    d22_out = diffusion(d22, L_mixer, w, U=i22 / (w * h), D=D)
    d23_out = diffusion(d23, L_mixer, w, U=i23 / (w * h), D=D)

    d20, d21, d22, d23 = d20_out, d21_out, d22_out, d23_out

    # Final stage

    ii30 = (V31 - V30) / Ri30
    ii31 = (V32 - V31) / Ri31
    ii32 = (V33 - V32) / Ri32
    ii33 = (V34 - V33) / Ri33
    ii34 = (V35 - V34) / Ri34
    ii35 = (V36 - V35) / Ri35

    i30 = V30 / R30
    i31 = V31 / R31
    i32 = V33 / R32
    i33 = V34 / R33
    i34 = V35 / R34

    d30, di30 = splitter(d_in=d20, q_l=i30, q_r=ii30, name="Fourth Stage - Left : d_in=d20, q_l=i30, q_r=ii30")
    di31, di32 = splitter(d_in=d21, q_l=ii31, q_r=ii32, name="Fourth Stage - Center 1 : d_in=d21, q_l=ii31, q_r=ii32")
    di33, di34 = splitter(d_in=d22, q_l=ii33, q_r=ii34, name="Fourth Stage - Center 2 : d_in=d22, q_l=ii33, q_r=ii34")
    di35, d34 = splitter(d_in=d23, q_l=ii35, q_r=i34, name="Fourth Stage - Right : d_in=d23, q_l=ii35, q_r=i34")

    # Petite diffusion dans le canal intermédiaire
    di30 = diffusion(di30, L_inter, w, U=ii30 / (w * h), D=D)
    di31 = diffusion(di31, L_inter, w, U=ii31 / (w * h), D=D)

    d31 = combiner(d_l=di30, d_r=di31, q_l=ii30, q_r=ii31,
                   name="Fourth Stage - Left : d_l=di30, d_r=di31, q_l=ii30, q_r=ii31")

    # Petite diffusion dans le canal intermédiaire
    di32 = diffusion(di32, L_inter, w, U=ii32 / (w * h), D=D)
    di33 = diffusion(di33, L_inter, w, U=ii33 / (w * h), D=D)

    d32 = combiner(d_l=di32, d_r=di33, q_l=ii32, q_r=ii33,
                   name="Fourth Stage - Center : d_l=di32, d_r=di33, q_l=ii32, q_r=ii33")

    # Petite diffusion dans le canal intermédiaire
    di34 = diffusion(di34, L_inter, w, U=ii33 / (w * h), D=D)
    di35 = diffusion(di35, L_inter, w, U=ii34 / (w * h), D=D)

    d33 = combiner(d_l=di34, d_r=di35, q_l=ii34, q_r=ii35,
                   name="Fourth Stage - Right : d_l=di34, d_r=di35, q_l=ii34, q_r=ii35")

    # diffusion dans les mixer
    d30_out = diffusion(d30, L_mixer, w, U=i30 / (w * h), D=D)
    d31_out = diffusion(d31, L_mixer, w, U=i31 / (w * h), D=D)
    d32_out = diffusion(d32, L_mixer, w, U=i32 / (w * h), D=D)
    d33_out = diffusion(d33, L_mixer, w, U=i33 / (w * h), D=D)
    d34_out = diffusion(d34, L_mixer, w, U=i34 / (w * h), D=D)
    if is_debug:
        print("First Stage")
        print("VA, VB : ", VA, VB)
        print("i00, i01 : ", i00, i01)
        print("")

        print("Second Stage")
        print("V10, V11, V12 : ", V10, V11, V12)
        print("ii10, ii11 : ", ii10, ii11)
        print("i10, i11, i12 : ", i10, i11, i12)
        print("d10_out[0], d11_out[0], d12_out[0] : ", d10_out[0], d11_out[0], d12_out[0])
        print("")

        print("Third Stage")
        print("V20, V21, V22, V23, V24 :", V20, V21, V22, V23, V24)
        print("ii20, ii21, ii22, ii23 :", ii20, ii21, ii22, ii23)
        print("i20, i21, i22, i23 :", i20, i21, i22, i23)
        print("d20_out[0], d21_out[0], d22_out[0], d23_out[0]] : ", d20_out[0], d21_out[0], d22_out[0], d23_out[0])
        print("")

        print("Fourth Stage")
        print("V30, V31, V32, V33, V34, V35, V36 :", V30, V31, V32, V33, V34, V35, V36)
        print("ii30, ii31, ii32, ii33, ii34, ii35 :", ii30, ii31, ii32, ii33, ii34, ii35)
        print("i30, i31, i32, i33, i34 :", i30, i31, i32, i33, i34)
        print("d30_out[0], d31_out[0], d32_out[0], d33_out[0], d34_out[0] : ", d30_out[0], d31_out[0], d32_out[0],
              d33_out[0], d34_out[0])
        print("")

    return np.array([d30_out[0], d31_out[0], d32_out[0], d33_out[0], d34_out[0]])


nb_exp_per_sigma = 150
sigmas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
sigmas = [0.1, 0.1, 0.1, 0.1, 0.1]
mean_scores = []
stds = []
relative_errors = []

for sigma in sigmas:
    print("sigma :", sigma)
    scores = []
    outputs_for_mean = []
    for n in range(nb_exp_per_sigma):
        outputs = calculate_output_gradient(sigma)
        outputs_for_mean.append(outputs)
        scores.append(np.sum(abs(outputs - [1, 0.75, 0.5, 0.25, 0])))
    means = np.mean(outputs_for_mean, axis=0)
    stds_ = np.std(outputs_for_mean, axis=0)

    print("means : ", means)
    relative_error = stds_ / means * 100
    print(relative_error)
    relative_errors.append([relative_error[1], relative_error[2], relative_error[3]])
    mean_scores.append(np.mean(scores))
    stds.append(np.std(scores))

print(relative_errors)

# print("mean_scores ; ", mean_scores)
# print("stds ; ", stds)
plt.errorbar(sigmas, mean_scores, yerr=stds)
plt.show()



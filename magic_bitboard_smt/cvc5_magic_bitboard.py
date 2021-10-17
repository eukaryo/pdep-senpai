# coding: UTF-8

# Author: Hiroki Takizawa, 2021

# License: MIT License

# Copyright(c) 2021 Hiroki Takizawa
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import re
import sys
import datetime

import pycvc5

def pdep(a: int, mask: int) -> int:
    dest = 0
    k = 0
    for m in range(64):
        if (mask & (1 << m)) != 0:
            if (a & (1 << k)) != 0:
                dest += 1 << m
            k += 1
    return dest

def popcount(n):
    return bin(n).count("1")

def bin64(n):
    return "0b"+bin(n)[2:].zfill(64)

def hex64(n):
    return "0x"+hex(n)[2:].zfill(16)

def solve(mask):

    slv = pycvc5.Solver()

    slv.setLogic("QF_ALL")
    slv.setOption("produce-models", "true")
    slv.setOption("output-language", "smt2")


    bitvector64 = slv.mkBitVectorSort(64)
    bitvector_ext = slv.mkOp(pycvc5.kinds.BVExtract, 63, 63 - popcount(mask) + 1)
    bitvector_short = slv.mkBitVectorSort(popcount(mask))
    set_ = slv.mkSetSort(slv.mkBitVectorSort(popcount(mask)))

    shift32 = slv.mkBitVector(64, 32)

    # masked_bb = [slv.mkBitVector(64, pdep(i, mask)) for i in range(2 ** popcount(mask))]
    masked_bb_u = [slv.mkTerm(pycvc5.kinds.BVShl, slv.mkBitVector(64, pdep(i, mask) // (2 ** 32)), shift32) for i in range(2 ** popcount(mask))]
    masked_bb_l = [slv.mkBitVector(64, pdep(i, mask) % (2 ** 32)) for i in range(2 ** popcount(mask))]
    masked_bb = [slv.mkTerm(pycvc5.kinds.BVAdd, masked_bb_u[i], masked_bb_l[i])  for i in range(2 ** popcount(mask))]

    magic_number = slv.mkConst(bitvector64, "magic_number")
    imul_tmp = [slv.mkTerm(pycvc5.kinds.BVMult, masked_bb[i], magic_number) for i in range(2 ** popcount(mask))]
    result_index_short = [slv.mkTerm(pycvc5.kinds.Singleton, slv.mkTerm(bitvector_ext, imul_tmp[i])) for i in range(2 ** popcount(mask))]

    target = [slv.mkTerm(pycvc5.kinds.Singleton, slv.mkBitVector(popcount(mask), i)) for i in range(2 ** popcount(mask))]

    union1 = [slv.mkEmptySet(set_)]
    union2 = [slv.mkEmptySet(set_)]
    for i in range(2 ** popcount(mask)):
        union1.append(slv.mkTerm(pycvc5.kinds.Union, union1[i - 1], result_index_short[i]))
        union2.append(slv.mkTerm(pycvc5.kinds.Union, union2[i - 1], target[i]))

    magic = slv.mkTerm(pycvc5.kinds.Equal, union1[-1], union2[-1])

    print("solve start")
    # print(f"{str(magic)}")

    result = slv.checkSatAssuming(magic)

    print(f"cvc5 reports: magic is {result}")

    if result:
        print(f"For instance, {slv.getValue(magic_number)} is a magic_number.")

def rookBlockMaskCalc(square):
    assert type(square) is int
    assert 0 <= square and square < 81
    FILE = square // 9
    RANK = square % 9
    FILE_BB = (0x1ff << (9 * FILE), 0)
    if FILE >= 7:
        FILE_BB = (0, 0x1ff << (9 * (FILE - 7)))
    RANK_BB = (0x40201008040201 << RANK, 0x201 << RANK)

    result = [FILE_BB[i] ^ RANK_BB[i] for i in range(2)]

    if FILE != 8:
        FILE9_BB = (0, 0x1ff << (9 * 1))
        result = [result[i] & ~FILE9_BB[i] for i in range(2)]
    if FILE != 0:
        FILE1_BB = (0x1ff << (9 * 0), 0)
        result = [result[i] & ~FILE1_BB[i] for i in range(2)]
    if RANK != 8:
        RANK9_BB = (0x40201008040201 << 8, 0x201 << 8)
        result = [result[i] & ~RANK9_BB[i] for i in range(2)]
    if RANK != 0:
        RANK1_BB = (0x40201008040201 << 0, 0x201 << 0)
        result = [result[i] & ~RANK1_BB[i] for i in range(2)]

    return result

def find_magic_number_rook(square):
    bb = rookBlockMaskCalc(square)
    print(f"square = {square}")
    print(f"bb[0] = {bin64(bb[0])}")
    print(f"bb[1] = {bin64(bb[1])}")
    assert (bb[0] & bb[1]) == 0
    mask = bb[0] | bb[1]
    pop = popcount(mask)
    print(f"start : {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : square = {square}, mask = {hex64(mask)} (pop = {popcount(mask)})")
    solve(bb[0] | bb[1])
    print(f"finish: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : square = {square}, mask = {hex64(mask)} (pop = {popcount(mask)})")


if __name__ == "__main__":

    # testmask = 0b1000001001100100110001
    # testmask = 0b10000010000000001110000000000
    # print(f"start : {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : mask = {hex64(testmask)} (pop = {popcount(testmask)})")
    # solve(testmask)
    # print(f"finish: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : mask = {hex64(testmask)} (pop = {popcount(testmask)})")

    find_magic_number_rook(17)
    find_magic_number_rook(53)

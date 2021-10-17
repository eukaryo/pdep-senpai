# coding: UTF-8

# Author: Hiroki Takizawa, 2021

# License: MIT License

# Copyright(c) 2021 Hiroki Takizawa
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import datetime

import z3
# z3をimportする簡単な方法：
# https://qiita.com/SatoshiTerasaki/items/476c9938479a4bfdda52

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

def solve(mask, compromise):

    shiftlen = 64 - popcount(mask) - compromise

    bb = z3.BitVec("bb", 64)
    masked_bb = [z3.BitVec(f"masked_bb_{i}", 64) for i in range(2 ** popcount(mask))]
    magic_number = z3.BitVec("magic_number", 64)
    imul_tmp = [z3.BitVec(f"imul_tmp_{i}", 64) for i in range(2 ** popcount(mask))]
    result_index_short = [z3.BitVec(f"result_index_{i}", popcount(mask) + compromise) for i in range(2 ** popcount(mask))]
    
    s = z3.Solver()
    for i in range(2 ** popcount(mask)):
        s.add(masked_bb[i] == pdep(i, mask))
        s.add(imul_tmp[i] == masked_bb[i] * magic_number)
        s.add(result_index_short[i] == z3.Extract(63, 63 - popcount(mask) - compromise + 1, imul_tmp[i]))
    s.add(z3.Distinct(result_index_short))
    # for i in range(2 ** popcount(mask)):
    #     for j in range(i + 1, 2 ** popcount(mask)):
    #         s.add(result_index_short[i] != result_index_short[j])
    s.add(magic_number >= 1)
    time_start = time.time()
    result = s.check()
    time_end = time.time()

    print(f"mask = {hex64(mask)}")
    print(f"compromise = {compromise}")
    print(f"result = {result}")
    print(f"elapsed time = {int(time_end - time_start)} second")

    if result == z3.unsat:
        return False

    print(f"magic_number = {hex64(s.model()[magic_number].as_long())}")

    return True

def rookBlockMaskCalc(square):
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

def find_magic_number_rook(square, compromise):
    bb = rookBlockMaskCalc(square)
    print(f"square = {square}")
    print(f"bb[0] = {bin64(bb[0])}")
    print(f"bb[1] = {bin64(bb[1])}")
    assert (bb[0] & bb[1]) == 0
    mask = bb[0] | bb[1]
    pop = popcount(mask)
    print(f"start : {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : square = {square}, mask = {hex64(mask)} (pop = {popcount(mask)}), compromise = {compromise}")
    solve(bb[0] | bb[1], compromise)
    print(f"finish: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : square = {square}, mask = {hex64(mask)} (pop = {popcount(mask)}), compromise = {compromise}")


if __name__ == "__main__":

    # solve(0b100000100000000011100000, 0)

    find_magic_number_rook(17, 0)
    find_magic_number_rook(53, 0)

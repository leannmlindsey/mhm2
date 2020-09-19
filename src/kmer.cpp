/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#include "kmer.hpp"

const uint64_t TWIN_TABLE[256] = {
    0xFF, 0xBF, 0x7F, 0x3F, 0xEF, 0xAF, 0x6F, 0x2F,
    0xDF, 0x9F, 0x5F, 0x1F, 0xCF, 0x8F, 0x4F, 0x0F,
    0xFB, 0xBB, 0x7B, 0x3B, 0xEB, 0xAB, 0x6B, 0x2B,
    0xDB, 0x9B, 0x5B, 0x1B, 0xCB, 0x8B, 0x4B, 0x0B,
    0xF7, 0xB7, 0x77, 0x37, 0xE7, 0xA7, 0x67, 0x27,
    0xD7, 0x97, 0x57, 0x17, 0xC7, 0x87, 0x47, 0x07,
    0xF3, 0xB3, 0x73, 0x33, 0xE3, 0xA3, 0x63, 0x23,
    0xD3, 0x93, 0x53, 0x13, 0xC3, 0x83, 0x43, 0x03,
    0xFE, 0xBE, 0x7E, 0x3E, 0xEE, 0xAE, 0x6E, 0x2E,
    0xDE, 0x9E, 0x5E, 0x1E, 0xCE, 0x8E, 0x4E, 0x0E,
    0xFA, 0xBA, 0x7A, 0x3A, 0xEA, 0xAA, 0x6A, 0x2A,
    0xDA, 0x9A, 0x5A, 0x1A, 0xCA, 0x8A, 0x4A, 0x0A,
    0xF6, 0xB6, 0x76, 0x36, 0xE6, 0xA6, 0x66, 0x26,
    0xD6, 0x96, 0x56, 0x16, 0xC6, 0x86, 0x46, 0x06,
    0xF2, 0xB2, 0x72, 0x32, 0xE2, 0xA2, 0x62, 0x22,
    0xD2, 0x92, 0x52, 0x12, 0xC2, 0x82, 0x42, 0x02,
    0xFD, 0xBD, 0x7D, 0x3D, 0xED, 0xAD, 0x6D, 0x2D,
    0xDD, 0x9D, 0x5D, 0x1D, 0xCD, 0x8D, 0x4D, 0x0D,
    0xF9, 0xB9, 0x79, 0x39, 0xE9, 0xA9, 0x69, 0x29,
    0xD9, 0x99, 0x59, 0x19, 0xC9, 0x89, 0x49, 0x09,
    0xF5, 0xB5, 0x75, 0x35, 0xE5, 0xA5, 0x65, 0x25,
    0xD5, 0x95, 0x55, 0x15, 0xC5, 0x85, 0x45, 0x05,
    0xF1, 0xB1, 0x71, 0x31, 0xE1, 0xA1, 0x61, 0x21,
    0xD1, 0x91, 0x51, 0x11, 0xC1, 0x81, 0x41, 0x01,
    0xFC, 0xBC, 0x7C, 0x3C, 0xEC, 0xAC, 0x6C, 0x2C,
    0xDC, 0x9C, 0x5C, 0x1C, 0xCC, 0x8C, 0x4C, 0x0C,
    0xF8, 0xB8, 0x78, 0x38, 0xE8, 0xA8, 0x68, 0x28,
    0xD8, 0x98, 0x58, 0x18, 0xC8, 0x88, 0x48, 0x08,
    0xF4, 0xB4, 0x74, 0x34, 0xE4, 0xA4, 0x64, 0x24,
    0xD4, 0x94, 0x54, 0x14, 0xC4, 0x84, 0x44, 0x04,
    0xF0, 0xB0, 0x70, 0x30, 0xE0, 0xA0, 0x60, 0x20,
    0xD0, 0x90, 0x50, 0x10, 0xC0, 0x80, 0x40, 0x00
  };


__MACRO_KMER__(32,  template);


#if MAX_BUILD_KMER >= 64

__MACRO_KMER__(64,  template);

#endif
#if MAX_BUILD_KMER >= 96

__MACRO_KMER__(96,  template);

#endif
#if MAX_BUILD_KMER >= 128

__MACRO_KMER__(128,  template);

#endif
#if MAX_BUILD_KMER >= 160

__MACRO_KMER__(160,  template);

#endif

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

.hsa_code_object_version 2, 1
.hsa_code_object_isa 8, 0, 3, "AMD", "AMDGPU"

.text
.globl ConvFwd1x1
.p2align 8
.type ConvFwd1x1,@function
.amdgpu_hsa_kernel ConvFwd1x1

ConvFwd1x1:
    .amd_kernel_code_t
        amd_code_version_major = 1
        amd_code_version_minor = 1
        amd_machine_kind = 1
        amd_machine_version_major = 8
        amd_machine_version_minor = 0
        amd_machine_version_stepping = 3
        kernarg_segment_alignment = 4
        group_segment_alignment = 4
        private_segment_alignment = 4
        wavefront_size = 6
        call_convention = -1
        enable_sgpr_kernarg_segment_ptr = 1
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_sgpr_workgroup_id_z = 1
        enable_vgpr_workitem_id = 2
        is_ptr64 = 1
        float_mode = 192
        granulated_wavefront_sgpr_count = 5
        granulated_workitem_vgpr_count = 8
        user_sgpr_count = 2
        wavefront_sgpr_count = 43
        workitem_vgpr_count = 36
        kernarg_segment_byte_size = 44
        workgroup_group_segment_byte_size = 0
    .end_amd_kernel_code_t
    
START_PROG:
    s_load_dwordx2                              s[6:7], s[0:1], 0
    s_load_dwordx2                              s[8:9], s[0:1], 8
    s_load_dwordx2                              s[10:11], s[0:1], 16
    s_load_dwordx2                              s[12:13], s[0:1], 24
    s_load_dwordx2                              s[14:15], s[0:1], 32
    s_load_dword                                s[5], s[0:1], 40
    s_lshl_b32                                  s[20], s[2], 2                           
    v_lshrrev_b32                               v[16], 6, v[0]                           
    v_add_u32                                   v[2], vcc, v[16], s[20]                  
    v_and_b32                                   v[3], 63, v[0]                           
    v_lshrrev_b32                               v[17], 7, v[2]                           
    v_lshrrev_b32                               v[4], 1, v[17]                           
    v_and_b32                                   v[5], 1, v[17]                           
    v_and_b32                                   v[6], 127, v[2]                          
    v_lshlrev_b32                               v[16], 6, v[4]                           
    v_add_u32                                   v[16], vcc, v[3], v[16]                  
    v_mov_b32                                   v[17], 49
    v_cvt_f32_u32                               v[8], v[16]
    v_mov_b32                                   v[7], 0.100000
    v_add_f32                                   v[8], v[8], v[7]                         
    v_cvt_f32_u32                               v[7], v[17]
    v_rcp_f32                                   v[7], v[7]
    v_mul_f32                                   v[7], v[8], v[7]                         
    v_cvt_u32_f32                               v[8], v[7]
    v_mul_u32_u24                               v[7], v[8], v[17]                        
    v_sub_u32                                   v[7], vcc, v[16], v[7]                   
    v_lshlrev_b32                               v[9], 2, v[6]                            
    v_mov_b32                                   v[16], 2
    v_cmpx_lt_u32                               vcc, v[8], v[16]                         
    s_cbranch_execz                             END_PROG
    v_mov_b32                                   v[16], 100352
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_lshlrev_b32                               v[17], 10, v[5]                          
    v_mov_b32                                   v[18], 49
    v_mul_u32_u24                               v[17], v[17], v[18]                      
    v_add_u32                                   v[18], vcc, v[16], v[17]                 
    v_addc_u32                                  v[18], vcc, v[18], v[7], vcc
    v_lshlrev_b32                               v[18], 2, v[18]                          
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[11], s[7]
    v_add_u32                                   v[10], vcc, s[6], v[18]                  
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    v_mov_b32                                   v[16], 2048
    v_mul_u32_u24                               v[16], v[9], v[16]                       
    v_lshlrev_b32                               v[17], 10, v[5]                          
    v_add_u32                                   v[16], vcc, v[16], v[17]                 
    v_readfirstlane_b32                         s[20], v[16]
    s_lshl_b32                                  s[20], s[20], 2                          
    s_waitcnt                                   lgkmcnt(0)
    s_add_u32                                   s[16], s[8], s[20]                       
    s_addc_u32                                  s[17], 0, s[9]                           
    v_lshlrev_b32                               v[16], 2, v[9]                           
    v_mov_b32                                   v[13], s[11]
    v_add_u32                                   v[12], vcc, s[10], v[16]                 
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    v_mov_b32                                   v[17], 128
    v_mul_u32_u24                               v[16], v[4], v[17]                       
    v_add_u32                                   v[16], vcc, v[16], v[6]                  
    v_readfirstlane_b32                         s[20], v[16]
    s_lshl_b32                                  s[20], s[20], 2                          
    s_waitcnt                                   lgkmcnt(0)
    s_add_u32                                   s[18], s[14], s[20]                      
    s_addc_u32                                  s[19], 0, s[15]                          
    v_mov_b32                                   v[16], 25088
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_mov_b32                                   v[17], 49
    v_mul_u32_u24                               v[17], v[9], v[17]                       
    v_add_u32                                   v[18], vcc, v[16], v[17]                 
    v_addc_u32                                  v[18], vcc, v[18], v[7], vcc
    v_lshlrev_b32                               v[18], 2, v[18]                          
    v_mov_b32                                   v[15], s[13]
    v_add_u32                                   v[14], vcc, s[12], v[18]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    v_mov_b32                                   v[32], 0
    v_mov_b32                                   v[33], 0
    v_mov_b32                                   v[34], 0
    v_mov_b32                                   v[35], 0
    v_readfirstlane_b32                         s[6], v[5]
    s_cmpk_eq_i32                               s[6], 0
    s_cbranch_scc0                              SEG_2_2
    s_mov_b32                                   s[7], 100
    s_store_dword                               s[7], s[18:19], 0                        glc
SEG_2_2:
    s_cmpk_eq_i32                               s[6], 0
    s_cbranch_scc0                              SEG_2
    v_mov_b32                                   v[6], v[14]
    v_mov_b32                                   v[7], v[15]
    v_mov_b32                                   v[2], 0
    flat_store_dword                            v[14:15], v[2]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_store_dword                            v[14:15], v[2]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_store_dword                            v[14:15], v[2]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_store_dword                            v[14:15], v[2]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[32], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[33], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[34], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[35], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    v_mov_b32                                   v[14], v[6]
    v_mov_b32                                   v[15], v[7]
SEG_2:
    s_waitcnt                                   0
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 8192
    s_load_dword                                s[34], s[16:17], 16384
    s_load_dword                                s[35], s[16:17], 24576
    flat_load_dword                             v[16], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_mov_b32                                   s[6], 63
CONV_LOOP:
    flat_load_dword                             v[24], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[25], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[26], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[27], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[28], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[29], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[30], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[31], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 8192
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[32], v[16], s[8]                       
    v_mac_f32                                   v[32], v[17], s[9]                       
    v_mac_f32                                   v[32], v[18], s[10]                      
    v_mac_f32                                   v[32], v[19], s[11]                      
    v_mac_f32                                   v[32], v[20], s[12]                      
    v_mac_f32                                   v[32], v[21], s[13]                      
    v_mac_f32                                   v[32], v[22], s[14]                      
    v_mac_f32                                   v[32], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 16384
    v_mac_f32                                   v[33], v[16], s[24]                      
    v_mac_f32                                   v[33], v[17], s[25]                      
    v_mac_f32                                   v[33], v[18], s[26]                      
    v_mac_f32                                   v[33], v[19], s[27]                      
    v_mac_f32                                   v[33], v[20], s[28]                      
    v_mac_f32                                   v[33], v[21], s[29]                      
    v_mac_f32                                   v[33], v[22], s[30]                      
    v_mac_f32                                   v[33], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 24576
    v_mac_f32                                   v[34], v[16], s[8]                       
    v_mac_f32                                   v[34], v[17], s[9]                       
    v_mac_f32                                   v[34], v[18], s[10]                      
    v_mac_f32                                   v[34], v[19], s[11]                      
    v_mac_f32                                   v[34], v[20], s[12]                      
    v_mac_f32                                   v[34], v[21], s[13]                      
    v_mac_f32                                   v[34], v[22], s[14]                      
    v_mac_f32                                   v[34], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[35], v[16], s[24]                      
    v_mac_f32                                   v[35], v[17], s[25]                      
    v_mac_f32                                   v[35], v[18], s[26]                      
    v_mac_f32                                   v[35], v[19], s[27]                      
    v_mac_f32                                   v[35], v[20], s[28]                      
    v_mac_f32                                   v[35], v[21], s[29]                      
    v_mac_f32                                   v[35], v[22], s[30]                      
    v_mac_f32                                   v[35], v[23], s[31]                      
    flat_load_dword                             v[16], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 8224
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[32], v[24], s[8]                       
    v_mac_f32                                   v[32], v[25], s[9]                       
    v_mac_f32                                   v[32], v[26], s[10]                      
    v_mac_f32                                   v[32], v[27], s[11]                      
    v_mac_f32                                   v[32], v[28], s[12]                      
    v_mac_f32                                   v[32], v[29], s[13]                      
    v_mac_f32                                   v[32], v[30], s[14]                      
    v_mac_f32                                   v[32], v[31], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 16416
    v_mac_f32                                   v[33], v[24], s[24]                      
    v_mac_f32                                   v[33], v[25], s[25]                      
    v_mac_f32                                   v[33], v[26], s[26]                      
    v_mac_f32                                   v[33], v[27], s[27]                      
    v_mac_f32                                   v[33], v[28], s[28]                      
    v_mac_f32                                   v[33], v[29], s[29]                      
    v_mac_f32                                   v[33], v[30], s[30]                      
    v_mac_f32                                   v[33], v[31], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 24608
    v_mac_f32                                   v[34], v[24], s[8]                       
    v_mac_f32                                   v[34], v[25], s[9]                       
    v_mac_f32                                   v[34], v[26], s[10]                      
    v_mac_f32                                   v[34], v[27], s[11]                      
    v_mac_f32                                   v[34], v[28], s[12]                      
    v_mac_f32                                   v[34], v[29], s[13]                      
    v_mac_f32                                   v[34], v[30], s[14]                      
    v_mac_f32                                   v[34], v[31], s[15]                      
    s_add_u32                                   s[16], s[16], 64                         
    s_addc_u32                                  s[17], s[17], 0                          
    s_waitcnt                                   lgkmcnt(0)
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 8192
    s_load_dword                                s[34], s[16:17], 16384
    s_load_dword                                s[35], s[16:17], 24576
    v_mac_f32                                   v[35], v[24], s[24]                      
    v_mac_f32                                   v[35], v[25], s[25]                      
    v_mac_f32                                   v[35], v[26], s[26]                      
    v_mac_f32                                   v[35], v[27], s[27]                      
    v_mac_f32                                   v[35], v[28], s[28]                      
    v_mac_f32                                   v[35], v[29], s[29]                      
    v_mac_f32                                   v[35], v[30], s[30]                      
    v_mac_f32                                   v[35], v[31], s[31]                      
    s_sub_u32                                   s[6], s[6], 1                            
    s_cmpk_eq_i32                               s[6], 0
    s_cbranch_scc0                              CONV_LOOP
    flat_load_dword                             v[24], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[25], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[26], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[27], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[28], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[29], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[30], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[31], v[10:11]                           
    v_add_u32                                   v[10], vcc, 196, v[10]                   
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 8192
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[32], v[16], s[8]                       
    v_mac_f32                                   v[32], v[17], s[9]                       
    v_mac_f32                                   v[32], v[18], s[10]                      
    v_mac_f32                                   v[32], v[19], s[11]                      
    v_mac_f32                                   v[32], v[20], s[12]                      
    v_mac_f32                                   v[32], v[21], s[13]                      
    v_mac_f32                                   v[32], v[22], s[14]                      
    v_mac_f32                                   v[32], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 16384
    v_mac_f32                                   v[33], v[16], s[24]                      
    v_mac_f32                                   v[33], v[17], s[25]                      
    v_mac_f32                                   v[33], v[18], s[26]                      
    v_mac_f32                                   v[33], v[19], s[27]                      
    v_mac_f32                                   v[33], v[20], s[28]                      
    v_mac_f32                                   v[33], v[21], s[29]                      
    v_mac_f32                                   v[33], v[22], s[30]                      
    v_mac_f32                                   v[33], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 24576
    v_mac_f32                                   v[34], v[16], s[8]                       
    v_mac_f32                                   v[34], v[17], s[9]                       
    v_mac_f32                                   v[34], v[18], s[10]                      
    v_mac_f32                                   v[34], v[19], s[11]                      
    v_mac_f32                                   v[34], v[20], s[12]                      
    v_mac_f32                                   v[34], v[21], s[13]                      
    v_mac_f32                                   v[34], v[22], s[14]                      
    v_mac_f32                                   v[34], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[35], v[16], s[24]                      
    v_mac_f32                                   v[35], v[17], s[25]                      
    v_mac_f32                                   v[35], v[18], s[26]                      
    v_mac_f32                                   v[35], v[19], s[27]                      
    v_mac_f32                                   v[35], v[20], s[28]                      
    v_mac_f32                                   v[35], v[21], s[29]                      
    v_mac_f32                                   v[35], v[22], s[30]                      
    v_mac_f32                                   v[35], v[23], s[31]                      
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 8224
    s_waitcnt                                   vmcnt(0)
    v_mac_f32                                   v[32], v[24], s[8]                       
    v_mac_f32                                   v[32], v[25], s[9]                       
    v_mac_f32                                   v[32], v[26], s[10]                      
    v_mac_f32                                   v[32], v[27], s[11]                      
    v_mac_f32                                   v[32], v[28], s[12]                      
    v_mac_f32                                   v[32], v[29], s[13]                      
    v_mac_f32                                   v[32], v[30], s[14]                      
    v_mac_f32                                   v[32], v[31], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 16416
    v_mac_f32                                   v[33], v[24], s[24]                      
    v_mac_f32                                   v[33], v[25], s[25]                      
    v_mac_f32                                   v[33], v[26], s[26]                      
    v_mac_f32                                   v[33], v[27], s[27]                      
    v_mac_f32                                   v[33], v[28], s[28]                      
    v_mac_f32                                   v[33], v[29], s[29]                      
    v_mac_f32                                   v[33], v[30], s[30]                      
    v_mac_f32                                   v[33], v[31], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 24608
    v_mac_f32                                   v[34], v[24], s[8]                       
    v_mac_f32                                   v[34], v[25], s[9]                       
    v_mac_f32                                   v[34], v[26], s[10]                      
    v_mac_f32                                   v[34], v[27], s[11]                      
    v_mac_f32                                   v[34], v[28], s[12]                      
    v_mac_f32                                   v[34], v[29], s[13]                      
    v_mac_f32                                   v[34], v[30], s[14]                      
    v_mac_f32                                   v[34], v[31], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[35], v[24], s[24]                      
    v_mac_f32                                   v[35], v[25], s[25]                      
    v_mac_f32                                   v[35], v[26], s[26]                      
    v_mac_f32                                   v[35], v[27], s[27]                      
    v_mac_f32                                   v[35], v[28], s[28]                      
    v_mac_f32                                   v[35], v[29], s[29]                      
    v_mac_f32                                   v[35], v[30], s[30]                      
    v_mac_f32                                   v[35], v[31], s[31]                      
    s_mov_b64                                   s[6:7], exec
    v_mov_b32                                   v[2], v[14]
    v_mov_b32                                   v[3], v[15]
    flat_load_dword                             v[7], v[14:15]                            glc
    s_waitcnt                                   vmcnt(0)
SEG_3_0:
    v_add_f32                                   v[6], v[7], v[32]                        
    flat_atomic_cmpswap                         v[4], v[14:15], v[6:7]                   glc
    s_waitcnt                                   vmcnt(0)
    v_cmpx_neq_f32                              vcc, v[7], v[4]                          
    v_mov_b32                                   v[7], v[4]
    s_cbranch_execnz                            SEG_3_0
    s_barrier
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[7], v[14:15]                            glc
    s_waitcnt                                   vmcnt(0)
SEG_3_1:
    v_add_f32                                   v[6], v[7], v[33]                        
    flat_atomic_cmpswap                         v[4], v[14:15], v[6:7]                   glc
    s_waitcnt                                   vmcnt(0)
    v_cmpx_neq_f32                              vcc, v[7], v[4]                          
    v_mov_b32                                   v[7], v[4]
    s_cbranch_execnz                            SEG_3_1
    s_barrier
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[7], v[14:15]                            glc
    s_waitcnt                                   vmcnt(0)
SEG_3_2:
    v_add_f32                                   v[6], v[7], v[34]                        
    flat_atomic_cmpswap                         v[4], v[14:15], v[6:7]                   glc
    s_waitcnt                                   vmcnt(0)
    v_cmpx_neq_f32                              vcc, v[7], v[4]                          
    v_mov_b32                                   v[7], v[4]
    s_cbranch_execnz                            SEG_3_2
    s_barrier
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[7], v[14:15]                            glc
    s_waitcnt                                   vmcnt(0)
SEG_3_3:
    v_add_f32                                   v[6], v[7], v[35]                        
    flat_atomic_cmpswap                         v[4], v[14:15], v[6:7]                   glc
    s_waitcnt                                   vmcnt(0)
    v_cmpx_neq_f32                              vcc, v[7], v[4]                          
    v_mov_b32                                   v[7], v[4]
    s_cbranch_execnz                            SEG_3_3
    s_barrier
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    v_mov_b32                                   v[6], s[18]
    v_mov_b32                                   v[7], s[19]
    s_mov_b64                                   exec, 1
    v_mov_b32                                   v[4], 1
    flat_atomic_add                             v[4], v[6:7], v[4]                       
    s_mov_b64                                   exec, s[6:7]
    s_waitcnt                                   vmcnt(0)
    v_readfirstlane_b32                         s[21], v[5]
    s_cmpk_eq_i32                               s[21], 1
    s_cbranch_scc0                              END_PROG
    s_mov_b32                                   s[22], 0
    s_mov_b32                                   s[23], 1000000
SEG_4:
    s_load_dword                                s[20], s[18:19], 0                       glc
    s_add_u32                                   s[22], s[22], 1                          
    s_cmp_eq_u32                                s[22], s[23]
    s_cbranch_scc1                              SEG_4_2
    s_waitcnt                                   lgkmcnt(0)
    s_cmpk_eq_i32                               s[20], 102
    s_cbranch_scc0                              SEG_4
SEG_4_2:
    s_mov_b64                                   exec, s[6:7]
    v_mov_b32                                   v[14], v[2]
    v_mov_b32                                   v[15], v[3]
    flat_load_dword                             v[32], v[14:15]                           glc
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[33], v[14:15]                           glc
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[34], v[14:15]                           glc
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    flat_load_dword                             v[35], v[14:15]                           glc
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    v_mov_b32                                   v[4], s[5]
    v_mov_b32                                   v[14], v[2]
    v_mov_b32                                   v[15], v[3]
    s_waitcnt                                   vmcnt(3)
    v_cmpx_lt_f32                               vcc, v[32], 0                            
    v_mul_f32                                   v[32], v[32], v[4]                       
    flat_store_dword                            v[14:15], v[32]
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_waitcnt                                   vmcnt(2)
    v_cmpx_lt_f32                               vcc, v[33], 0                            
    v_mul_f32                                   v[33], v[33], v[4]                       
    flat_store_dword                            v[14:15], v[33]
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_waitcnt                                   vmcnt(1)
    v_cmpx_lt_f32                               vcc, v[34], 0                            
    v_mul_f32                                   v[34], v[34], v[4]                       
    flat_store_dword                            v[14:15], v[34]
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_waitcnt                                   vmcnt(0)
    v_cmpx_lt_f32                               vcc, v[35], 0                            
    v_mul_f32                                   v[35], v[35], v[4]                       
    flat_store_dword                            v[14:15], v[35]
    s_mov_b64                                   exec, s[6:7]
    v_add_u32                                   v[14], vcc, 196, v[14]                   
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
END_PROG:
    s_endpgm

.amd_amdgpu_hsa_metadata
{ Version: [1, 0],
  Kernels :
    - { Name: ConvFwd1x1,
        SymbolName: ConvFwd1x1,
        Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs: { ReqdWorkGroupSize: [ 256, 1, 1 ] }
        CodeProps: { KernargSegmentSize: 44, GroupSegmentFixedSize : 0, PrivateSegmentFixedSize : 0, KernargSegmentAlign : 8, WavefrontSize : 64, MaxFlatWorkGroupSize : 256 }
        Args:
        - { Name: d_in  , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_wei , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_bias , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
        - { Name: d_out , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
        - { Name: d_sig , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global  }
        - { Name: d_nSlop , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, IsConst: true }
      }
}
.end_amd_amdgpu_hsa_metadata


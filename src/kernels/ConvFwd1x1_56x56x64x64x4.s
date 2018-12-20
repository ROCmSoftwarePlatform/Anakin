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
        granulated_wavefront_sgpr_count = 6
        granulated_workitem_vgpr_count = 11
        user_sgpr_count = 2
        wavefront_sgpr_count = 55
        workitem_vgpr_count = 48
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
    v_lshrrev_b32                               v[17], 2, v[2]                           
    v_lshrrev_b32                               v[4], 0, v[17]                           
    v_and_b32                                   v[5], 0, v[17]                           
    v_and_b32                                   v[6], 3, v[2]                            
    v_lshlrev_b32                               v[16], 6, v[4]                           
    v_add_u32                                   v[16], vcc, v[3], v[16]                  
    v_mov_b32                                   v[17], 3136
    v_cvt_f32_u32                               v[8], v[16]
    v_mov_b32                                   v[7], 0.100000
    v_add_f32                                   v[8], v[8], v[7]                         
    v_cvt_f32_u32                               v[7], v[17]
    v_rcp_f32                                   v[7], v[7]
    v_mul_f32                                   v[7], v[8], v[7]                         
    v_cvt_u32_f32                               v[8], v[7]
    v_mul_u32_u24                               v[7], v[8], v[17]                        
    v_sub_u32                                   v[7], vcc, v[16], v[7]                   
    v_lshlrev_b32                               v[9], 4, v[6]                            
    v_mov_b32                                   v[16], 4
    v_cmpx_lt_u32                               vcc, v[8], v[16]                         
    s_cbranch_execz                             END_PROG
    v_mov_b32                                   v[16], 200704
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_lshlrev_b32                               v[17], 6, v[5]                           
    v_mov_b32                                   v[18], 3136
    v_mul_u32_u24                               v[17], v[17], v[18]                      
    v_add_u32                                   v[18], vcc, v[16], v[17]                 
    v_addc_u32                                  v[18], vcc, v[18], v[7], vcc
    v_lshlrev_b32                               v[18], 2, v[18]                          
    s_waitcnt                                   lgkmcnt(0)
    v_mov_b32                                   v[11], s[7]
    v_add_u32                                   v[10], vcc, s[6], v[18]                  
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    v_mov_b32                                   v[16], 64
    v_mul_u32_u24                               v[16], v[9], v[16]                       
    v_lshlrev_b32                               v[17], 6, v[5]                           
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
    v_mov_b32                                   v[17], 4
    v_mul_u32_u24                               v[16], v[4], v[17]                       
    v_add_u32                                   v[16], vcc, v[16], v[6]                  
    v_readfirstlane_b32                         s[20], v[16]
    s_lshl_b32                                  s[20], s[20], 2                          
    s_waitcnt                                   lgkmcnt(0)
    s_add_u32                                   s[18], s[14], s[20]                      
    s_addc_u32                                  s[19], 0, s[15]                          
    v_mov_b32                                   v[16], 200704
    v_mul_u32_u24                               v[16], v[8], v[16]                       
    v_mov_b32                                   v[17], 3136
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
    v_mov_b32                                   v[36], 0
    v_mov_b32                                   v[37], 0
    v_mov_b32                                   v[38], 0
    v_mov_b32                                   v[39], 0
    v_mov_b32                                   v[40], 0
    v_mov_b32                                   v[41], 0
    v_mov_b32                                   v[42], 0
    v_mov_b32                                   v[43], 0
    v_mov_b32                                   v[44], 0
    v_mov_b32                                   v[45], 0
    v_mov_b32                                   v[46], 0
    v_mov_b32                                   v[47], 0
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
    flat_load_dword                             v[36], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[37], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[38], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[39], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[40], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[41], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[42], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[43], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[44], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[45], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[46], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    flat_load_dword                             v[47], v[12:13]                           
    v_add_u32                                   v[12], vcc, 4, v[12]                     
    v_addc_u32                                  v[13], vcc, 0, v[13], vcc
    s_waitcnt                                   0
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 256
    s_load_dword                                s[34], s[16:17], 512
    s_load_dword                                s[35], s[16:17], 768
    s_load_dword                                s[36], s[16:17], 1024
    s_load_dword                                s[37], s[16:17], 1280
    s_load_dword                                s[38], s[16:17], 1536
    s_load_dword                                s[39], s[16:17], 1792
    s_load_dword                                s[40], s[16:17], 2048
    s_load_dword                                s[41], s[16:17], 2304
    s_load_dword                                s[42], s[16:17], 2560
    s_load_dword                                s[43], s[16:17], 2816
    s_load_dword                                s[44], s[16:17], 3072
    s_load_dword                                s[45], s[16:17], 3328
    s_load_dword                                s[46], s[16:17], 3584
    s_load_dword                                s[47], s[16:17], 3840
    flat_load_dword                             v[2], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[3], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[4], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[5], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[6], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[7], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[8], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[9], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_mov_b32                                   s[6], 3
CONV_LOOP:
    flat_load_dword                             v[16], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 256
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[32], v[2], s[8]                        
    v_mac_f32                                   v[32], v[3], s[9]                        
    v_mac_f32                                   v[32], v[4], s[10]                       
    v_mac_f32                                   v[32], v[5], s[11]                       
    v_mac_f32                                   v[32], v[6], s[12]                       
    v_mac_f32                                   v[32], v[7], s[13]                       
    v_mac_f32                                   v[32], v[8], s[14]                       
    v_mac_f32                                   v[32], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 512
    v_mac_f32                                   v[33], v[2], s[24]                       
    v_mac_f32                                   v[33], v[3], s[25]                       
    v_mac_f32                                   v[33], v[4], s[26]                       
    v_mac_f32                                   v[33], v[5], s[27]                       
    v_mac_f32                                   v[33], v[6], s[28]                       
    v_mac_f32                                   v[33], v[7], s[29]                       
    v_mac_f32                                   v[33], v[8], s[30]                       
    v_mac_f32                                   v[33], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 768
    v_mac_f32                                   v[34], v[2], s[8]                        
    v_mac_f32                                   v[34], v[3], s[9]                        
    v_mac_f32                                   v[34], v[4], s[10]                       
    v_mac_f32                                   v[34], v[5], s[11]                       
    v_mac_f32                                   v[34], v[6], s[12]                       
    v_mac_f32                                   v[34], v[7], s[13]                       
    v_mac_f32                                   v[34], v[8], s[14]                       
    v_mac_f32                                   v[34], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1024
    v_mac_f32                                   v[35], v[2], s[24]                       
    v_mac_f32                                   v[35], v[3], s[25]                       
    v_mac_f32                                   v[35], v[4], s[26]                       
    v_mac_f32                                   v[35], v[5], s[27]                       
    v_mac_f32                                   v[35], v[6], s[28]                       
    v_mac_f32                                   v[35], v[7], s[29]                       
    v_mac_f32                                   v[35], v[8], s[30]                       
    v_mac_f32                                   v[35], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1280
    v_mac_f32                                   v[36], v[2], s[8]                        
    v_mac_f32                                   v[36], v[3], s[9]                        
    v_mac_f32                                   v[36], v[4], s[10]                       
    v_mac_f32                                   v[36], v[5], s[11]                       
    v_mac_f32                                   v[36], v[6], s[12]                       
    v_mac_f32                                   v[36], v[7], s[13]                       
    v_mac_f32                                   v[36], v[8], s[14]                       
    v_mac_f32                                   v[36], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1536
    v_mac_f32                                   v[37], v[2], s[24]                       
    v_mac_f32                                   v[37], v[3], s[25]                       
    v_mac_f32                                   v[37], v[4], s[26]                       
    v_mac_f32                                   v[37], v[5], s[27]                       
    v_mac_f32                                   v[37], v[6], s[28]                       
    v_mac_f32                                   v[37], v[7], s[29]                       
    v_mac_f32                                   v[37], v[8], s[30]                       
    v_mac_f32                                   v[37], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1792
    v_mac_f32                                   v[38], v[2], s[8]                        
    v_mac_f32                                   v[38], v[3], s[9]                        
    v_mac_f32                                   v[38], v[4], s[10]                       
    v_mac_f32                                   v[38], v[5], s[11]                       
    v_mac_f32                                   v[38], v[6], s[12]                       
    v_mac_f32                                   v[38], v[7], s[13]                       
    v_mac_f32                                   v[38], v[8], s[14]                       
    v_mac_f32                                   v[38], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2048
    v_mac_f32                                   v[39], v[2], s[24]                       
    v_mac_f32                                   v[39], v[3], s[25]                       
    v_mac_f32                                   v[39], v[4], s[26]                       
    v_mac_f32                                   v[39], v[5], s[27]                       
    v_mac_f32                                   v[39], v[6], s[28]                       
    v_mac_f32                                   v[39], v[7], s[29]                       
    v_mac_f32                                   v[39], v[8], s[30]                       
    v_mac_f32                                   v[39], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2304
    v_mac_f32                                   v[40], v[2], s[8]                        
    v_mac_f32                                   v[40], v[3], s[9]                        
    v_mac_f32                                   v[40], v[4], s[10]                       
    v_mac_f32                                   v[40], v[5], s[11]                       
    v_mac_f32                                   v[40], v[6], s[12]                       
    v_mac_f32                                   v[40], v[7], s[13]                       
    v_mac_f32                                   v[40], v[8], s[14]                       
    v_mac_f32                                   v[40], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2560
    v_mac_f32                                   v[41], v[2], s[24]                       
    v_mac_f32                                   v[41], v[3], s[25]                       
    v_mac_f32                                   v[41], v[4], s[26]                       
    v_mac_f32                                   v[41], v[5], s[27]                       
    v_mac_f32                                   v[41], v[6], s[28]                       
    v_mac_f32                                   v[41], v[7], s[29]                       
    v_mac_f32                                   v[41], v[8], s[30]                       
    v_mac_f32                                   v[41], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2816
    v_mac_f32                                   v[42], v[2], s[8]                        
    v_mac_f32                                   v[42], v[3], s[9]                        
    v_mac_f32                                   v[42], v[4], s[10]                       
    v_mac_f32                                   v[42], v[5], s[11]                       
    v_mac_f32                                   v[42], v[6], s[12]                       
    v_mac_f32                                   v[42], v[7], s[13]                       
    v_mac_f32                                   v[42], v[8], s[14]                       
    v_mac_f32                                   v[42], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3072
    v_mac_f32                                   v[43], v[2], s[24]                       
    v_mac_f32                                   v[43], v[3], s[25]                       
    v_mac_f32                                   v[43], v[4], s[26]                       
    v_mac_f32                                   v[43], v[5], s[27]                       
    v_mac_f32                                   v[43], v[6], s[28]                       
    v_mac_f32                                   v[43], v[7], s[29]                       
    v_mac_f32                                   v[43], v[8], s[30]                       
    v_mac_f32                                   v[43], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3328
    v_mac_f32                                   v[44], v[2], s[8]                        
    v_mac_f32                                   v[44], v[3], s[9]                        
    v_mac_f32                                   v[44], v[4], s[10]                       
    v_mac_f32                                   v[44], v[5], s[11]                       
    v_mac_f32                                   v[44], v[6], s[12]                       
    v_mac_f32                                   v[44], v[7], s[13]                       
    v_mac_f32                                   v[44], v[8], s[14]                       
    v_mac_f32                                   v[44], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3584
    v_mac_f32                                   v[45], v[2], s[24]                       
    v_mac_f32                                   v[45], v[3], s[25]                       
    v_mac_f32                                   v[45], v[4], s[26]                       
    v_mac_f32                                   v[45], v[5], s[27]                       
    v_mac_f32                                   v[45], v[6], s[28]                       
    v_mac_f32                                   v[45], v[7], s[29]                       
    v_mac_f32                                   v[45], v[8], s[30]                       
    v_mac_f32                                   v[45], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3840
    v_mac_f32                                   v[46], v[2], s[8]                        
    v_mac_f32                                   v[46], v[3], s[9]                        
    v_mac_f32                                   v[46], v[4], s[10]                       
    v_mac_f32                                   v[46], v[5], s[11]                       
    v_mac_f32                                   v[46], v[6], s[12]                       
    v_mac_f32                                   v[46], v[7], s[13]                       
    v_mac_f32                                   v[46], v[8], s[14]                       
    v_mac_f32                                   v[46], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[47], v[2], s[24]                       
    v_mac_f32                                   v[47], v[3], s[25]                       
    v_mac_f32                                   v[47], v[4], s[26]                       
    v_mac_f32                                   v[47], v[5], s[27]                       
    v_mac_f32                                   v[47], v[6], s[28]                       
    v_mac_f32                                   v[47], v[7], s[29]                       
    v_mac_f32                                   v[47], v[8], s[30]                       
    v_mac_f32                                   v[47], v[9], s[31]                       
    flat_load_dword                             v[2], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[3], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[4], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[5], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[6], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[7], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[8], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[9], v[10:11]                            
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 288
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
    s_load_dwordx8                              s[8:15], s[16:17], 544
    v_mac_f32                                   v[33], v[16], s[24]                      
    v_mac_f32                                   v[33], v[17], s[25]                      
    v_mac_f32                                   v[33], v[18], s[26]                      
    v_mac_f32                                   v[33], v[19], s[27]                      
    v_mac_f32                                   v[33], v[20], s[28]                      
    v_mac_f32                                   v[33], v[21], s[29]                      
    v_mac_f32                                   v[33], v[22], s[30]                      
    v_mac_f32                                   v[33], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 800
    v_mac_f32                                   v[34], v[16], s[8]                       
    v_mac_f32                                   v[34], v[17], s[9]                       
    v_mac_f32                                   v[34], v[18], s[10]                      
    v_mac_f32                                   v[34], v[19], s[11]                      
    v_mac_f32                                   v[34], v[20], s[12]                      
    v_mac_f32                                   v[34], v[21], s[13]                      
    v_mac_f32                                   v[34], v[22], s[14]                      
    v_mac_f32                                   v[34], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1056
    v_mac_f32                                   v[35], v[16], s[24]                      
    v_mac_f32                                   v[35], v[17], s[25]                      
    v_mac_f32                                   v[35], v[18], s[26]                      
    v_mac_f32                                   v[35], v[19], s[27]                      
    v_mac_f32                                   v[35], v[20], s[28]                      
    v_mac_f32                                   v[35], v[21], s[29]                      
    v_mac_f32                                   v[35], v[22], s[30]                      
    v_mac_f32                                   v[35], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1312
    v_mac_f32                                   v[36], v[16], s[8]                       
    v_mac_f32                                   v[36], v[17], s[9]                       
    v_mac_f32                                   v[36], v[18], s[10]                      
    v_mac_f32                                   v[36], v[19], s[11]                      
    v_mac_f32                                   v[36], v[20], s[12]                      
    v_mac_f32                                   v[36], v[21], s[13]                      
    v_mac_f32                                   v[36], v[22], s[14]                      
    v_mac_f32                                   v[36], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1568
    v_mac_f32                                   v[37], v[16], s[24]                      
    v_mac_f32                                   v[37], v[17], s[25]                      
    v_mac_f32                                   v[37], v[18], s[26]                      
    v_mac_f32                                   v[37], v[19], s[27]                      
    v_mac_f32                                   v[37], v[20], s[28]                      
    v_mac_f32                                   v[37], v[21], s[29]                      
    v_mac_f32                                   v[37], v[22], s[30]                      
    v_mac_f32                                   v[37], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1824
    v_mac_f32                                   v[38], v[16], s[8]                       
    v_mac_f32                                   v[38], v[17], s[9]                       
    v_mac_f32                                   v[38], v[18], s[10]                      
    v_mac_f32                                   v[38], v[19], s[11]                      
    v_mac_f32                                   v[38], v[20], s[12]                      
    v_mac_f32                                   v[38], v[21], s[13]                      
    v_mac_f32                                   v[38], v[22], s[14]                      
    v_mac_f32                                   v[38], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2080
    v_mac_f32                                   v[39], v[16], s[24]                      
    v_mac_f32                                   v[39], v[17], s[25]                      
    v_mac_f32                                   v[39], v[18], s[26]                      
    v_mac_f32                                   v[39], v[19], s[27]                      
    v_mac_f32                                   v[39], v[20], s[28]                      
    v_mac_f32                                   v[39], v[21], s[29]                      
    v_mac_f32                                   v[39], v[22], s[30]                      
    v_mac_f32                                   v[39], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2336
    v_mac_f32                                   v[40], v[16], s[8]                       
    v_mac_f32                                   v[40], v[17], s[9]                       
    v_mac_f32                                   v[40], v[18], s[10]                      
    v_mac_f32                                   v[40], v[19], s[11]                      
    v_mac_f32                                   v[40], v[20], s[12]                      
    v_mac_f32                                   v[40], v[21], s[13]                      
    v_mac_f32                                   v[40], v[22], s[14]                      
    v_mac_f32                                   v[40], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2592
    v_mac_f32                                   v[41], v[16], s[24]                      
    v_mac_f32                                   v[41], v[17], s[25]                      
    v_mac_f32                                   v[41], v[18], s[26]                      
    v_mac_f32                                   v[41], v[19], s[27]                      
    v_mac_f32                                   v[41], v[20], s[28]                      
    v_mac_f32                                   v[41], v[21], s[29]                      
    v_mac_f32                                   v[41], v[22], s[30]                      
    v_mac_f32                                   v[41], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2848
    v_mac_f32                                   v[42], v[16], s[8]                       
    v_mac_f32                                   v[42], v[17], s[9]                       
    v_mac_f32                                   v[42], v[18], s[10]                      
    v_mac_f32                                   v[42], v[19], s[11]                      
    v_mac_f32                                   v[42], v[20], s[12]                      
    v_mac_f32                                   v[42], v[21], s[13]                      
    v_mac_f32                                   v[42], v[22], s[14]                      
    v_mac_f32                                   v[42], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3104
    v_mac_f32                                   v[43], v[16], s[24]                      
    v_mac_f32                                   v[43], v[17], s[25]                      
    v_mac_f32                                   v[43], v[18], s[26]                      
    v_mac_f32                                   v[43], v[19], s[27]                      
    v_mac_f32                                   v[43], v[20], s[28]                      
    v_mac_f32                                   v[43], v[21], s[29]                      
    v_mac_f32                                   v[43], v[22], s[30]                      
    v_mac_f32                                   v[43], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3360
    v_mac_f32                                   v[44], v[16], s[8]                       
    v_mac_f32                                   v[44], v[17], s[9]                       
    v_mac_f32                                   v[44], v[18], s[10]                      
    v_mac_f32                                   v[44], v[19], s[11]                      
    v_mac_f32                                   v[44], v[20], s[12]                      
    v_mac_f32                                   v[44], v[21], s[13]                      
    v_mac_f32                                   v[44], v[22], s[14]                      
    v_mac_f32                                   v[44], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3616
    v_mac_f32                                   v[45], v[16], s[24]                      
    v_mac_f32                                   v[45], v[17], s[25]                      
    v_mac_f32                                   v[45], v[18], s[26]                      
    v_mac_f32                                   v[45], v[19], s[27]                      
    v_mac_f32                                   v[45], v[20], s[28]                      
    v_mac_f32                                   v[45], v[21], s[29]                      
    v_mac_f32                                   v[45], v[22], s[30]                      
    v_mac_f32                                   v[45], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3872
    v_mac_f32                                   v[46], v[16], s[8]                       
    v_mac_f32                                   v[46], v[17], s[9]                       
    v_mac_f32                                   v[46], v[18], s[10]                      
    v_mac_f32                                   v[46], v[19], s[11]                      
    v_mac_f32                                   v[46], v[20], s[12]                      
    v_mac_f32                                   v[46], v[21], s[13]                      
    v_mac_f32                                   v[46], v[22], s[14]                      
    v_mac_f32                                   v[46], v[23], s[15]                      
    s_add_u32                                   s[16], s[16], 64                         
    s_addc_u32                                  s[17], s[17], 0                          
    s_waitcnt                                   lgkmcnt(0)
    s_load_dword                                s[32], s[16:17], 0
    s_load_dword                                s[33], s[16:17], 256
    s_load_dword                                s[34], s[16:17], 512
    s_load_dword                                s[35], s[16:17], 768
    s_load_dword                                s[36], s[16:17], 1024
    s_load_dword                                s[37], s[16:17], 1280
    s_load_dword                                s[38], s[16:17], 1536
    s_load_dword                                s[39], s[16:17], 1792
    s_load_dword                                s[40], s[16:17], 2048
    s_load_dword                                s[41], s[16:17], 2304
    s_load_dword                                s[42], s[16:17], 2560
    s_load_dword                                s[43], s[16:17], 2816
    s_load_dword                                s[44], s[16:17], 3072
    s_load_dword                                s[45], s[16:17], 3328
    s_load_dword                                s[46], s[16:17], 3584
    s_load_dword                                s[47], s[16:17], 3840
    v_mac_f32                                   v[47], v[16], s[24]                      
    v_mac_f32                                   v[47], v[17], s[25]                      
    v_mac_f32                                   v[47], v[18], s[26]                      
    v_mac_f32                                   v[47], v[19], s[27]                      
    v_mac_f32                                   v[47], v[20], s[28]                      
    v_mac_f32                                   v[47], v[21], s[29]                      
    v_mac_f32                                   v[47], v[22], s[30]                      
    v_mac_f32                                   v[47], v[23], s[31]                      
    s_sub_u32                                   s[6], s[6], 1                            
    s_cmpk_eq_i32                               s[6], 0
    s_cbranch_scc0                              CONV_LOOP
    flat_load_dword                             v[16], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[17], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[18], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[19], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[20], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[21], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[22], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    flat_load_dword                             v[23], v[10:11]                           
    v_add_u32                                   v[10], vcc, 12544, v[10]                 
    v_addc_u32                                  v[11], vcc, 0, v[11], vcc
    s_load_dwordx8                              s[8:15], s[16:17], 0
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 256
    s_waitcnt                                   vmcnt(8)
    v_mac_f32                                   v[32], v[2], s[8]                        
    v_mac_f32                                   v[32], v[3], s[9]                        
    v_mac_f32                                   v[32], v[4], s[10]                       
    v_mac_f32                                   v[32], v[5], s[11]                       
    v_mac_f32                                   v[32], v[6], s[12]                       
    v_mac_f32                                   v[32], v[7], s[13]                       
    v_mac_f32                                   v[32], v[8], s[14]                       
    v_mac_f32                                   v[32], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 512
    v_mac_f32                                   v[33], v[2], s[24]                       
    v_mac_f32                                   v[33], v[3], s[25]                       
    v_mac_f32                                   v[33], v[4], s[26]                       
    v_mac_f32                                   v[33], v[5], s[27]                       
    v_mac_f32                                   v[33], v[6], s[28]                       
    v_mac_f32                                   v[33], v[7], s[29]                       
    v_mac_f32                                   v[33], v[8], s[30]                       
    v_mac_f32                                   v[33], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 768
    v_mac_f32                                   v[34], v[2], s[8]                        
    v_mac_f32                                   v[34], v[3], s[9]                        
    v_mac_f32                                   v[34], v[4], s[10]                       
    v_mac_f32                                   v[34], v[5], s[11]                       
    v_mac_f32                                   v[34], v[6], s[12]                       
    v_mac_f32                                   v[34], v[7], s[13]                       
    v_mac_f32                                   v[34], v[8], s[14]                       
    v_mac_f32                                   v[34], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1024
    v_mac_f32                                   v[35], v[2], s[24]                       
    v_mac_f32                                   v[35], v[3], s[25]                       
    v_mac_f32                                   v[35], v[4], s[26]                       
    v_mac_f32                                   v[35], v[5], s[27]                       
    v_mac_f32                                   v[35], v[6], s[28]                       
    v_mac_f32                                   v[35], v[7], s[29]                       
    v_mac_f32                                   v[35], v[8], s[30]                       
    v_mac_f32                                   v[35], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1280
    v_mac_f32                                   v[36], v[2], s[8]                        
    v_mac_f32                                   v[36], v[3], s[9]                        
    v_mac_f32                                   v[36], v[4], s[10]                       
    v_mac_f32                                   v[36], v[5], s[11]                       
    v_mac_f32                                   v[36], v[6], s[12]                       
    v_mac_f32                                   v[36], v[7], s[13]                       
    v_mac_f32                                   v[36], v[8], s[14]                       
    v_mac_f32                                   v[36], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1536
    v_mac_f32                                   v[37], v[2], s[24]                       
    v_mac_f32                                   v[37], v[3], s[25]                       
    v_mac_f32                                   v[37], v[4], s[26]                       
    v_mac_f32                                   v[37], v[5], s[27]                       
    v_mac_f32                                   v[37], v[6], s[28]                       
    v_mac_f32                                   v[37], v[7], s[29]                       
    v_mac_f32                                   v[37], v[8], s[30]                       
    v_mac_f32                                   v[37], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1792
    v_mac_f32                                   v[38], v[2], s[8]                        
    v_mac_f32                                   v[38], v[3], s[9]                        
    v_mac_f32                                   v[38], v[4], s[10]                       
    v_mac_f32                                   v[38], v[5], s[11]                       
    v_mac_f32                                   v[38], v[6], s[12]                       
    v_mac_f32                                   v[38], v[7], s[13]                       
    v_mac_f32                                   v[38], v[8], s[14]                       
    v_mac_f32                                   v[38], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2048
    v_mac_f32                                   v[39], v[2], s[24]                       
    v_mac_f32                                   v[39], v[3], s[25]                       
    v_mac_f32                                   v[39], v[4], s[26]                       
    v_mac_f32                                   v[39], v[5], s[27]                       
    v_mac_f32                                   v[39], v[6], s[28]                       
    v_mac_f32                                   v[39], v[7], s[29]                       
    v_mac_f32                                   v[39], v[8], s[30]                       
    v_mac_f32                                   v[39], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2304
    v_mac_f32                                   v[40], v[2], s[8]                        
    v_mac_f32                                   v[40], v[3], s[9]                        
    v_mac_f32                                   v[40], v[4], s[10]                       
    v_mac_f32                                   v[40], v[5], s[11]                       
    v_mac_f32                                   v[40], v[6], s[12]                       
    v_mac_f32                                   v[40], v[7], s[13]                       
    v_mac_f32                                   v[40], v[8], s[14]                       
    v_mac_f32                                   v[40], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2560
    v_mac_f32                                   v[41], v[2], s[24]                       
    v_mac_f32                                   v[41], v[3], s[25]                       
    v_mac_f32                                   v[41], v[4], s[26]                       
    v_mac_f32                                   v[41], v[5], s[27]                       
    v_mac_f32                                   v[41], v[6], s[28]                       
    v_mac_f32                                   v[41], v[7], s[29]                       
    v_mac_f32                                   v[41], v[8], s[30]                       
    v_mac_f32                                   v[41], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2816
    v_mac_f32                                   v[42], v[2], s[8]                        
    v_mac_f32                                   v[42], v[3], s[9]                        
    v_mac_f32                                   v[42], v[4], s[10]                       
    v_mac_f32                                   v[42], v[5], s[11]                       
    v_mac_f32                                   v[42], v[6], s[12]                       
    v_mac_f32                                   v[42], v[7], s[13]                       
    v_mac_f32                                   v[42], v[8], s[14]                       
    v_mac_f32                                   v[42], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3072
    v_mac_f32                                   v[43], v[2], s[24]                       
    v_mac_f32                                   v[43], v[3], s[25]                       
    v_mac_f32                                   v[43], v[4], s[26]                       
    v_mac_f32                                   v[43], v[5], s[27]                       
    v_mac_f32                                   v[43], v[6], s[28]                       
    v_mac_f32                                   v[43], v[7], s[29]                       
    v_mac_f32                                   v[43], v[8], s[30]                       
    v_mac_f32                                   v[43], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3328
    v_mac_f32                                   v[44], v[2], s[8]                        
    v_mac_f32                                   v[44], v[3], s[9]                        
    v_mac_f32                                   v[44], v[4], s[10]                       
    v_mac_f32                                   v[44], v[5], s[11]                       
    v_mac_f32                                   v[44], v[6], s[12]                       
    v_mac_f32                                   v[44], v[7], s[13]                       
    v_mac_f32                                   v[44], v[8], s[14]                       
    v_mac_f32                                   v[44], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3584
    v_mac_f32                                   v[45], v[2], s[24]                       
    v_mac_f32                                   v[45], v[3], s[25]                       
    v_mac_f32                                   v[45], v[4], s[26]                       
    v_mac_f32                                   v[45], v[5], s[27]                       
    v_mac_f32                                   v[45], v[6], s[28]                       
    v_mac_f32                                   v[45], v[7], s[29]                       
    v_mac_f32                                   v[45], v[8], s[30]                       
    v_mac_f32                                   v[45], v[9], s[31]                       
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3840
    v_mac_f32                                   v[46], v[2], s[8]                        
    v_mac_f32                                   v[46], v[3], s[9]                        
    v_mac_f32                                   v[46], v[4], s[10]                       
    v_mac_f32                                   v[46], v[5], s[11]                       
    v_mac_f32                                   v[46], v[6], s[12]                       
    v_mac_f32                                   v[46], v[7], s[13]                       
    v_mac_f32                                   v[46], v[8], s[14]                       
    v_mac_f32                                   v[46], v[9], s[15]                       
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[47], v[2], s[24]                       
    v_mac_f32                                   v[47], v[3], s[25]                       
    v_mac_f32                                   v[47], v[4], s[26]                       
    v_mac_f32                                   v[47], v[5], s[27]                       
    v_mac_f32                                   v[47], v[6], s[28]                       
    v_mac_f32                                   v[47], v[7], s[29]                       
    v_mac_f32                                   v[47], v[8], s[30]                       
    v_mac_f32                                   v[47], v[9], s[31]                       
    s_load_dwordx8                              s[8:15], s[16:17], 32
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 288
    s_waitcnt                                   vmcnt(0)
    v_mac_f32                                   v[32], v[16], s[8]                       
    v_mac_f32                                   v[32], v[17], s[9]                       
    v_mac_f32                                   v[32], v[18], s[10]                      
    v_mac_f32                                   v[32], v[19], s[11]                      
    v_mac_f32                                   v[32], v[20], s[12]                      
    v_mac_f32                                   v[32], v[21], s[13]                      
    v_mac_f32                                   v[32], v[22], s[14]                      
    v_mac_f32                                   v[32], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 544
    v_mac_f32                                   v[33], v[16], s[24]                      
    v_mac_f32                                   v[33], v[17], s[25]                      
    v_mac_f32                                   v[33], v[18], s[26]                      
    v_mac_f32                                   v[33], v[19], s[27]                      
    v_mac_f32                                   v[33], v[20], s[28]                      
    v_mac_f32                                   v[33], v[21], s[29]                      
    v_mac_f32                                   v[33], v[22], s[30]                      
    v_mac_f32                                   v[33], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 800
    v_mac_f32                                   v[34], v[16], s[8]                       
    v_mac_f32                                   v[34], v[17], s[9]                       
    v_mac_f32                                   v[34], v[18], s[10]                      
    v_mac_f32                                   v[34], v[19], s[11]                      
    v_mac_f32                                   v[34], v[20], s[12]                      
    v_mac_f32                                   v[34], v[21], s[13]                      
    v_mac_f32                                   v[34], v[22], s[14]                      
    v_mac_f32                                   v[34], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1056
    v_mac_f32                                   v[35], v[16], s[24]                      
    v_mac_f32                                   v[35], v[17], s[25]                      
    v_mac_f32                                   v[35], v[18], s[26]                      
    v_mac_f32                                   v[35], v[19], s[27]                      
    v_mac_f32                                   v[35], v[20], s[28]                      
    v_mac_f32                                   v[35], v[21], s[29]                      
    v_mac_f32                                   v[35], v[22], s[30]                      
    v_mac_f32                                   v[35], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1312
    v_mac_f32                                   v[36], v[16], s[8]                       
    v_mac_f32                                   v[36], v[17], s[9]                       
    v_mac_f32                                   v[36], v[18], s[10]                      
    v_mac_f32                                   v[36], v[19], s[11]                      
    v_mac_f32                                   v[36], v[20], s[12]                      
    v_mac_f32                                   v[36], v[21], s[13]                      
    v_mac_f32                                   v[36], v[22], s[14]                      
    v_mac_f32                                   v[36], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 1568
    v_mac_f32                                   v[37], v[16], s[24]                      
    v_mac_f32                                   v[37], v[17], s[25]                      
    v_mac_f32                                   v[37], v[18], s[26]                      
    v_mac_f32                                   v[37], v[19], s[27]                      
    v_mac_f32                                   v[37], v[20], s[28]                      
    v_mac_f32                                   v[37], v[21], s[29]                      
    v_mac_f32                                   v[37], v[22], s[30]                      
    v_mac_f32                                   v[37], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 1824
    v_mac_f32                                   v[38], v[16], s[8]                       
    v_mac_f32                                   v[38], v[17], s[9]                       
    v_mac_f32                                   v[38], v[18], s[10]                      
    v_mac_f32                                   v[38], v[19], s[11]                      
    v_mac_f32                                   v[38], v[20], s[12]                      
    v_mac_f32                                   v[38], v[21], s[13]                      
    v_mac_f32                                   v[38], v[22], s[14]                      
    v_mac_f32                                   v[38], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2080
    v_mac_f32                                   v[39], v[16], s[24]                      
    v_mac_f32                                   v[39], v[17], s[25]                      
    v_mac_f32                                   v[39], v[18], s[26]                      
    v_mac_f32                                   v[39], v[19], s[27]                      
    v_mac_f32                                   v[39], v[20], s[28]                      
    v_mac_f32                                   v[39], v[21], s[29]                      
    v_mac_f32                                   v[39], v[22], s[30]                      
    v_mac_f32                                   v[39], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2336
    v_mac_f32                                   v[40], v[16], s[8]                       
    v_mac_f32                                   v[40], v[17], s[9]                       
    v_mac_f32                                   v[40], v[18], s[10]                      
    v_mac_f32                                   v[40], v[19], s[11]                      
    v_mac_f32                                   v[40], v[20], s[12]                      
    v_mac_f32                                   v[40], v[21], s[13]                      
    v_mac_f32                                   v[40], v[22], s[14]                      
    v_mac_f32                                   v[40], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 2592
    v_mac_f32                                   v[41], v[16], s[24]                      
    v_mac_f32                                   v[41], v[17], s[25]                      
    v_mac_f32                                   v[41], v[18], s[26]                      
    v_mac_f32                                   v[41], v[19], s[27]                      
    v_mac_f32                                   v[41], v[20], s[28]                      
    v_mac_f32                                   v[41], v[21], s[29]                      
    v_mac_f32                                   v[41], v[22], s[30]                      
    v_mac_f32                                   v[41], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 2848
    v_mac_f32                                   v[42], v[16], s[8]                       
    v_mac_f32                                   v[42], v[17], s[9]                       
    v_mac_f32                                   v[42], v[18], s[10]                      
    v_mac_f32                                   v[42], v[19], s[11]                      
    v_mac_f32                                   v[42], v[20], s[12]                      
    v_mac_f32                                   v[42], v[21], s[13]                      
    v_mac_f32                                   v[42], v[22], s[14]                      
    v_mac_f32                                   v[42], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3104
    v_mac_f32                                   v[43], v[16], s[24]                      
    v_mac_f32                                   v[43], v[17], s[25]                      
    v_mac_f32                                   v[43], v[18], s[26]                      
    v_mac_f32                                   v[43], v[19], s[27]                      
    v_mac_f32                                   v[43], v[20], s[28]                      
    v_mac_f32                                   v[43], v[21], s[29]                      
    v_mac_f32                                   v[43], v[22], s[30]                      
    v_mac_f32                                   v[43], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3360
    v_mac_f32                                   v[44], v[16], s[8]                       
    v_mac_f32                                   v[44], v[17], s[9]                       
    v_mac_f32                                   v[44], v[18], s[10]                      
    v_mac_f32                                   v[44], v[19], s[11]                      
    v_mac_f32                                   v[44], v[20], s[12]                      
    v_mac_f32                                   v[44], v[21], s[13]                      
    v_mac_f32                                   v[44], v[22], s[14]                      
    v_mac_f32                                   v[44], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[8:15], s[16:17], 3616
    v_mac_f32                                   v[45], v[16], s[24]                      
    v_mac_f32                                   v[45], v[17], s[25]                      
    v_mac_f32                                   v[45], v[18], s[26]                      
    v_mac_f32                                   v[45], v[19], s[27]                      
    v_mac_f32                                   v[45], v[20], s[28]                      
    v_mac_f32                                   v[45], v[21], s[29]                      
    v_mac_f32                                   v[45], v[22], s[30]                      
    v_mac_f32                                   v[45], v[23], s[31]                      
    s_waitcnt                                   lgkmcnt(0)
    s_load_dwordx8                              s[24:31], s[16:17], 3872
    v_mac_f32                                   v[46], v[16], s[8]                       
    v_mac_f32                                   v[46], v[17], s[9]                       
    v_mac_f32                                   v[46], v[18], s[10]                      
    v_mac_f32                                   v[46], v[19], s[11]                      
    v_mac_f32                                   v[46], v[20], s[12]                      
    v_mac_f32                                   v[46], v[21], s[13]                      
    v_mac_f32                                   v[46], v[22], s[14]                      
    v_mac_f32                                   v[46], v[23], s[15]                      
    s_waitcnt                                   lgkmcnt(0)
    v_mac_f32                                   v[47], v[16], s[24]                      
    v_mac_f32                                   v[47], v[17], s[25]                      
    v_mac_f32                                   v[47], v[18], s[26]                      
    v_mac_f32                                   v[47], v[19], s[27]                      
    v_mac_f32                                   v[47], v[20], s[28]                      
    v_mac_f32                                   v[47], v[21], s[29]                      
    v_mac_f32                                   v[47], v[22], s[30]                      
    v_mac_f32                                   v[47], v[23], s[31]                      
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[32], 0                            
    v_mul_f32                                   v[32], v[32], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[32]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[33], 0                            
    v_mul_f32                                   v[33], v[33], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[33]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[34], 0                            
    v_mul_f32                                   v[34], v[34], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[34]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[35], 0                            
    v_mul_f32                                   v[35], v[35], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[35]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[36], 0                            
    v_mul_f32                                   v[36], v[36], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[36]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[37], 0                            
    v_mul_f32                                   v[37], v[37], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[37]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[38], 0                            
    v_mul_f32                                   v[38], v[38], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[38]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[39], 0                            
    v_mul_f32                                   v[39], v[39], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[39]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[40], 0                            
    v_mul_f32                                   v[40], v[40], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[40]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[41], 0                            
    v_mul_f32                                   v[41], v[41], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[41]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[42], 0                            
    v_mul_f32                                   v[42], v[42], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[42]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[43], 0                            
    v_mul_f32                                   v[43], v[43], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[43]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[44], 0                            
    v_mul_f32                                   v[44], v[44], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[44]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[45], 0                            
    v_mul_f32                                   v[45], v[45], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[45]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[46], 0                            
    v_mul_f32                                   v[46], v[46], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[46]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
    v_addc_u32                                  v[15], vcc, 0, v[15], vcc
    s_mov_b64                                   s[6:7], exec
    v_cmpx_lt_f32                               vcc, v[47], 0                            
    v_mul_f32                                   v[47], v[47], s[5]                       
    s_mov_b64                                   exec, s[6:7]
    flat_store_dword                            v[14:15], v[47]
    v_add_u32                                   v[14], vcc, 12544, v[14]                 
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


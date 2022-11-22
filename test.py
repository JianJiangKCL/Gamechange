def get_BasicBlock_comb(dwpw):
    quantizer_dw = dwpw.quantizer_dw
    quantizer_pw = dwpw.quantizer_pw
    quantizer_dw2 = dwpw.quantizer_dw2
    quantizer_pw2 = dwpw.quantizer_pw2
    fq_dw = dwpw.feat_quantizer_dw.quantizer
    fq_pw = dwpw.feat_quantizer_pw.quantizer
    fq_dw2 = dwpw.feat_quantizer_dw2.quantizer
    fq_pw2 = dwpw.feat_quantizer_pw2.quantizer
    # get layer to pass its stride and padding
    dw_conv = dwpw.dw_conv
    pw_conv = dwpw.pw_conv
    dw_conv2 = dwpw.dw_conv2
    dw_comb_keys = direct_comb(fq_dw, quantizer_dw, fq_pw, dw_conv)

    pw_comb_keys = direct_comb(fq_pw, quantizer_pw, fq_dw2, pw_conv)

    dw2_comb_keys = direct_comb(fq_dw2, quantizer_dw2, fq_pw2, dw_conv2)
    return dw_comb_keys, pw_comb_keys, dw2_comb_keys
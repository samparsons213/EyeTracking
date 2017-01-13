function [d_ec_inv_d_ec] = dECInvdEC(ec_inv)
% Returns the derivative of emission_covs_inv(c, d, x) wrt
% emission_covs(a, b, x) for all a,b,c,d,x

    m = size(ec_inv, 3);
    d_ec_inv_d_ec = zeros(2, 2, 2, 2, m);
    for x_idx = 1:m
        for ec_idx1 = 1:2
            for ec_idx2 = 1:2
                for ec_inv_idx1 = 1:2
                    for ec_inv_idx2 = 1:2
                        d_ec_inv_d_ec(ec_inv_idx1, ec_inv_idx2,...
                            ec_idx1, ec_idx2, x_idx) = -ec_inv(ec_inv_idx1, ec_idx1, x_idx) *...
                            ec_inv(ec_idx2, ec_inv_idx2, x_idx);
                        if ec_idx2 ~= ec_idx1
                            d_ec_inv_d_ec(ec_inv_idx1, ec_inv_idx2,...
                                ec_idx1, ec_idx2, x_idx) =...
                                d_ec_inv_d_ec(ec_inv_idx1, ec_inv_idx2,...
                                ec_idx1, ec_idx2, x_idx) -...
                                ec_inv(ec_inv_idx1, ec_idx2, x_idx) *...
                                ec_inv(ec_idx1, ec_inv_idx2, x_idx);
                        end
                    end
                end
            end
        end
    end

end
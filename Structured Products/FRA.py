og_rate = .0325
notional = 25000000

small_rate = .0310
big_rate = .0345

small_rate_days = 40
big_rate_days = 130

fwd_rate = (1 + big_rate * big_rate_days/360) / (1 + small_rate * small_rate_days/360) -1

fwd_rate_ann = fwd_rate * 360 / (big_rate_days-small_rate_days)

print("Forward Rate (UnAnn) = "+ str(round(fwd_rate*100,2)) + "% (" + str(big_rate_days - small_rate_days) + " days)")

print("Forward Rate (Ann) = " + str(round(fwd_rate_ann*100,2))+ "%")

fv_gain =  (fwd_rate - og_rate * (big_rate_days-small_rate_days)/360 ) * notional
print("FV Gain/Loss = " + str(round(fv_gain,2)))

fra_gain = fv_gain / (1+big_rate*big_rate_days/360)

print("FV Gain/Loss = " + str(round(fra_gain,2)) + " (Discounted @" + str(big_rate_days) + "Days)")
og_rate = .04
notional = 10000000

small_rate = .0380
big_rate = .0410

small_rate_days = 60
big_rate_days = 240

fwd_rate = (1 + big_rate * big_rate_days/360) / (1 + small_rate * small_rate_days/360) -1

fwd_rate_ann = fwd_rate * 360 / (big_rate_days-small_rate_days)

print(fwd_rate)

print(str(fwd_rate_ann*100) + "%")

fv_gain =  (fwd_rate - og_rate * (big_rate_days-small_rate_days)/360 ) * notional
print(fv_gain)

fra_gain = fv_gain / (1+big_rate*big_rate_days/360)

print(fra_gain)
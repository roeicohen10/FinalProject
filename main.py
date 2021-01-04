from Streaming.Streaming import Stream_Data
from OL.OLModel import OLModel,MODELS
from Analyze.Analyze import Analyze
from alfha.onlineFDR_proc.AlphaInvest import ALPHA_proc

# def investing(X, Y, w, param):
#     a=ALPHA_proc(0.5)
#     return a.next_alpha(0),param



s=Stream_Data()
s.set_data("har.csv","target")
s.set_batch_size(50)
s.set_num_feature(-1)
s.set_ol(MODELS[0])
s.set_ofs()
s.params["w0"]=0.05
s.params["dw"]=0.05
s.prepare_data(0)
print(s.data.shape[1])

print("start simulate")
s.simulate_stream()
print("end simulate")
# s.get_plot_stats()
a = Analyze(s.stats)
a.show_accuracy_measures_plot()
a.show_time_measures_plot()
a.show_memory_measures_plot()
for f in s.stats["features"]:
    print(len(f))
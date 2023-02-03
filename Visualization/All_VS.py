import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class All_Visualization(object):
	"""docstring for ClassName"""
	def __init__(self, do_plot=True, save=False, dpi = 200):
		super(All_Visualization, self).__init__()
		self.do_plot = do_plot
		self.save = save
		self.dpi = dpi



	def boxplot_ELBM_SELBM(self, title, file_name_result):

		sns.set_theme(style="ticks")

		f, ax = plt.subplots(figsize=(5, 3.5))

		file_name_result = file_name_result + '.xlsx'
		df_box = pd.read_excel(file_name_result, header=None)
		df_box.columns = ['Algorithms','Measures','Values']
		ax = sns.boxplot(x = df_box['Values'], y=df_box['Algorithms'], hue = df_box['Measures'] , data=df_box, color="#3C4048" , width=0.5 ,whis=np.inf)
		#ax.xaxis.grid(True)
		plt.ylabel("Algorithms", rotation=90, fontname="ecbx1000")
		ax.tick_params(axis='y', rotation=0)

		ax.set_yticklabels([r"ELBM",r"SELBM"])
		plt.xlabel('Values', rotation=0, fontname="ecbx1000")
		#plt.title(title, y=-0.30, fontsize = 15, fontname="ecbx1000", fontweight="bold")
		ax.legend(bbox_to_anchor=(0,0))
		plt.grid()

		if self.do_plot:
			plt.show()

		if self.save :
			plt.savefig('file_name_result.pdf', format='pdf', dpi=self.dpi)

		pass

	def all_boxplot(self, list_title, list_file_name_result):

		sns.set_theme(style="ticks")

		f, ax = plt.subplots(figsize=(5, 3.5))
		for i in np.arange(1):

			plt.subplot(1, 2, i+1)
			self.boxplot_ELBM_SELBM(list_title[i], list_file_name_result[i])

		if self.do_plot:
			plt.show()

		if self.save :
			plt.savefig('all_boxplot_ELBM_SELBM.pdf', format='pdf', dpi=self.dpi)

		pass
		
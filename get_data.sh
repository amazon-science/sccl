#git clone https://github.com/rashadulrakib/short-text-clustering-enhancement.git

cd short-text-clustering-enhancement

predata=../pre_data

# create a new directory
mkdir -p $predata

# copy the raw data to it
cp data/agnewsdataraw-8000 $predata/agnews.csv
cp data/search_snippets/search_snippets_true_text $predata/search_snippets.csv
cp data/stackoverflow/stackoverflow_true_text $predata/stackoverflow.csv
cp data/biomedical/biomedical_true_text $predata/biomedical.csv
cp data/tweet-original-order.txt $predata/tweet.csv
cp data/S $predata/googlenews_S.csv
cp data/TS $predata/googlenews_TS.csv
cp data/T $predata/googlenews_T.csv

# go to it
cd $predata

# replace the tab char to "," comma for pandas
# but there exists "," in stackoverflow
#sed -i 's/\t/,/' *.csv
# stackoverflow has issues.
sed -i 's/\t/@@@/;s/\t/ /g;s/@@@/\t/' $predata/stackoverflow.csv

# insert "label,txt" as the first row for pandas
sed -i '1i\label\ttext' *.csv

cd ..
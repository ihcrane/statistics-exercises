#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Has the network latency gone up since we switched internet service providers?

# null hypothesis: There is no change in network latency since we switched internet providers
# alternate hypothesis: The network latency has gone up since we switched internet providers

# true positive: The network latency has increased since the switch to the new internet provider 
# true negative: The network latency has not increased since the switch to the new internet provider
# type I error: The network latency has not increased, however, the sample data suggests it has.
# type II error: The network latency has increased, however, the sample data suggests it did not.


# In[2]:


# 2. Is the website redesign any good?
# Rewording: Is the website redesign has increased sales.

# null hypothesis: The website redesign did not increase sales.
# alternate hypothesis: The website redesign increased sales.

# true positive: The website redesign increased sales
# true negative: The website redesign did not increase sales.
# type 1 error: The website redesign did not increase sales, however, the sample data suggests it did.
# type 2 error: The website redesign increased sales, however, the sample data suggests it did not.


# In[ ]:


# 3. Is our television ad driving more sales?

# null hypothesis: The television ad has not increased sales
# alternate hypothesis: The televesion ad has increased sales

# true positive: The televesion ad increased sales.
# true negative: The television ad did not increase sales.
# type 1 error: The television ad did not increase sales, however, the sample data suggests it did.
# type 2 error: The television ad increased sales, however, the sample data suggests it did not.


# ### Central Limit Theorem
# 
# - A population, that is not necessarily normally distributed.
# - Taking samples, of sufficient size, will result in normally distributed sample means.

# In[2]:


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


np.random.seed(123)

n_dice_per_experiment = ncols = 10
n_experiments = nrows = 100

data = np.random.randint(1,7, (nrows, ncols))

data[:4]


# In[7]:


calculated_averages = data.mean(axis=1)
calculated_averages


# In[8]:


plt.hist(calculated_averages)
plt.xlabel(f'Average of {n_dice_per_experiment} dice rolls')
plt.ylabel('# of Occurences')

plt.title(f'Out come of averaging {n_dice_per_experiment} dice rolls {n_experiments} times')


# In[20]:


import env

db_url = f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/telco_churn'
df = pd.read_sql('SELECT * FROM customers', db_url)  


# In[21]:


df.columns


# In[22]:


df.monthly_charges.hist()


# In[23]:


churn_sample = df[df.churn == 'Yes'].monthly_charges

churn_sample.hist()


# ## Set hypothesis
# 
# H_0: Mean of monthly charges of churned customers <= mean of monthly charges of all customers
# 
# H_a: Mean of monthly charges of churned customers > mean of monthly charges of all customers
#     

# In[29]:


## Set an alpha value

alpha = .05


# In[25]:


## Verify an assumption of sample size

df['churn'].value_counts()


# In[30]:


## Compute Test Statistic 


# Churned customer sample
churn_sample = df[df['churn'] == 'Yes']['monthly_charges']

# The overall mean
overall_mean = df['monthly_charges'].mean()


## The output of a 1 tail, 1 sample, t-test
# specified the sample
# specified the overall mean

# Assign the output to t (for the t-statistic) and p (the p-value)
t, p = stats.ttest_1samp(churn_sample, overall_mean)

print(t, p/2, alpha)


# In[32]:


# is t > 0

t > 0

# is p/2 < alpha

p/2 < alpha


# In[36]:


## The results based on t-statistic and p-value

if p/2 > alpha:
    print(f'We fail to reject null hypothesis')
elif t < 0:
    print(f'We fail to reject null hypothesis')
else:
    print('We reject null hypothesis')


# ## ANOVA - Analysis of Variance
# 
# Outcome: Compare means of groups A, B, and C
#     
# 1. Plot distributions
# 2. Estabilsh hypothesis
# 
# 
# ## Format of Hypothesis
# 
# $H_0$ Null hypothesis = $\mu_{A} = \mu_{B} = \mu_{C}$
# 
# $H_a$ Alternate hypothesis = $\mu_{A} \neq \mu_{B} \neq \mu_{C}$
# 
# ## Significant Level
# 
# alpha = .05
# 
# ## Verify assumptions
# 
# - Normal distribution or at least 30 observations
# - Indpendent variables
# - Equal variances
# 
# ### Syntax for test
# ```python
# scipy.stats.f_oneway
# ```
# Return: test statistitics and a p-value

# In[203]:


### Load a dataset

df = sns.load_dataset('iris')

### Check sample size
df. species.value_counts()


# df.info()

# Independent - belong to one species
# Sample size - sufficient


# In[ ]:


# Group A - setosa
# Group B - versicolor
# Group C - virginica


# In[204]:


# Statistical summary of sepal_length

df.sepal_length.describe()


# In[205]:


### Filter sepal_length by species

versicolor_sepal_length = df[df['species'] == 'versicolor']['sepal_length']
virginica_sepal_length = df[df['species'] == 'virginica']['sepal_length']
setosa_sepal_length = df[df['species'] == 'setosa']['sepal_length']


# In[209]:


## Histograms

versicolor_sepal_length.hist(alpha = .5)
setosa_sepal_length.hist(alpha = .5)
virginica_sepal_length.hist(alpha = .5)


# In[ ]:


### State hypotheses

H_0: For the mean of sepal_length - Versicolor = Virginica = Setosa
H_a: For the mean of sepal_length - Not all equal


# In[213]:


# significance level

alpha = .05

# Independence? YES
# 30 observations
# Equal variance


# Test for equal variance: Levene's Test
# 
# H_0: population variances of sepal length across all 3 species are equal
# H_a: population variances of sepal lengths is different among at least 2 species

# In[214]:


stats.levene(versicolor_sepal_length, setosa_sepal_length, virginica_sepal_length)


# The p-value is < .05 therefore the variance of at least 2 of the groups are significantly different. In this case, it would not be recommended to run an ANOVA because assumptions are violated.
# However, we will do it here to see what happens.
# In practice, we would resort to a non-parametric version of our test, which in the case of ANOVA, would be Kruskal-Wallis Test.

# In[215]:


stats.f_oneway(versicolor_sepal_length, virginica_sepal_length, setosa_sepal_length)


# In[216]:


stats.kruskal(versicolor_sepal_length, virginica_sepal_length, setosa_sepal_length)


# Because variances were not equal, we would go with the result of th Kruskal-Wallis Test.
# We still achieved significance here.
# 
# Our takeaway is that there appears to be a significant difference in sepal length across at least 2 of the groups.

# In[217]:


print(versicolor_sepal_length.mean())
print(virginica_sepal_length.mean())
print(setosa_sepal_length.mean())


# In[220]:


stats.ttest_ind(versicolor_sepal_length, virginica_sepal_length, equal_var=False)


# In[221]:


stats.ttest_ind(versicolor_sepal_length, setosa_sepal_length, equal_var= False)


# In comparing each group combination, we can see there is a significant difference in the Sepal Length of each species

# In[224]:


import seaborn as sns
mpg = sns.load_dataset('mpg')


# In[225]:


mpg.head()


# In[227]:


mpg = mpg[~mpg['horsepower'].isna()]
mpg.isna().sum()


# In[228]:


mpg['origin'].unique()


# Is the horsepower of vehicles different across the disctinct origins of the vehicles?
# 
# $H_{0}$: mean_horsepower_USA = mean_horsepower_japan == mean_horsepower_europe 
# 
# $H_{a}$: mean horsepower form at least 2 countries are significantly different 
# 
# - Assumptions
# 1. At least 30 observations: met(see unique above)
# 2. Equal population variance across all 3 groups
# 3. Groups are independent of each other

# In[231]:


mpg[mpg['origin']=='usa'].horsepower.hist()
mpg[mpg['origin']=='japan'].horsepower.hist()
mpg[mpg['origin']=='europe'].horsepower.hist()


# In[232]:


usa = mpg[mpg['origin']=='usa'].horsepower
japan = mpg[mpg['origin']=='japan'].horsepower
europe = mpg[mpg['origin']=='europe'].horsepower


# In[233]:


# validate variance assumption

stats.levene(usa, japan, europe)


# With a low p-value, we can assume the variances of the countries horsepower is significantly different across at least two of the countries
# With this information, we should not use the ANOVA test but should instead use the Krukal-Wallis Test

# In[236]:


t, p = stats.kruskal(usa, japan, europe)
p


# With a p-value < .05 (alpha), we can say there exists a significant difference between the horsepower of vehicles in at least 2 of the countries. But which 2?
# 
# Compare 2 groups, usa and japan
# 
# H_0: hp_usa == hp_japan
# 
# h_a: hp_usa != hp_japan

# In[244]:


# compare 2 groups to see if there is significance

t, p = stats.ttest_ind(usa, japan, equal_var=False)

if p < .05:
    print(f'There is a significant difference between the horsepower of vehicles in the USA vs Japan. (p-value: {p})')
else:
    print('We fail to find a signifcant difference')


# In[245]:


# compare 2 groups to see if there is significance

t, p = stats.ttest_ind(usa, europe, equal_var=False)

if p < .05:
    print(f'There is a significant difference between the horsepower of vehicles in the USA vs Europe. (p-value: {p})')
else:
    print('We fail to find a signifcant difference')


# In[247]:


# compare 2 groups to see if there is significance

t, p = stats.ttest_ind(japan, europe, equal_var=False)

if p < .05:
    print(f'There is a significant difference between the horsepower of vehicles in the Europe vs Japan. (p-value: {p})')
else:
    print(f'We fail to find a signifcant difference. (p-value: {p})')


# In[ ]:





# In[ ]:





# In[ ]:





# - Answer with the type of test you would use (assume normal distribution):
# 
# 1. Is there a difference in grades of students on the second floor compared to grades of all students?
#  - 1 sample t-test

# 2. Are adults who drink milk taller than adults who dont drink milk?
#  - independent t-test (two sample t-test)
#  - 1 tail

# 3. Is the price of gas higher in texas or in new mexico?
#  - indepndent t-test (two sample t-test)
#  - 1 tail

# 4. Are there differences in stress levels between students who take data science vs students who take web development vs students who take cloud academy?
#  - ANOVA

# #### 2. Ace Realty wants to determine whether the average time it takes to sell homes is different for its two offices. A sample of 40 sales from office #1 revealed a mean of 90 days and a standard deviation of 15 days. A sample of 50 sales from office #2 revealed a mean of 100 days and a standard deviation of 20 days. Use a .05 level of significance.

# In[7]:


np.random.seed(3)


# In[15]:


# Null Hypothesis: There is no significance betwen office 1 and office 2.
# Alternate hypothesis: Office 1 takes less time to sell homes than office 2.
# Alternate hypothesis: Office 2 takes less time to sell homes than office 1.

alpha = .05
off_1_mean = 90
off_1_sd = 15

off_2_mean = 100
off_2_sd = 20

off_1 = stats.norm(off_1_mean, off_2_sd).rvs(40)
plt.hist(off_1)


# In[16]:


off_2 = stats.norm(off_2_mean, off_2_sd).rvs(50)
plt.hist(off_2)


# In[17]:


print(off_1.var())
print(off_2.var())


# In[18]:


t, p = stats.ttest_ind(off_1, off_2, equal_var=False)
t, p/2


# In[24]:


# Null Hypothesis: There is no significance betwen office 1 and office 2.
# Alternate hypothesis: There is a significant difference between the offices.


# In[25]:


if p <= alpha:
    print("Reject Null")
else:
    print("Fail to reject null")


# #### 1. Is there a difference in fuel-efficiency in cars from 2008 vs 1999?
#  - independent t-test

# In[ ]:


# Null hypothesis: Mean of fuel-efficiency of 2008 cars <= Mean of fuel-efficiency of 1999 cars.
# Alternate Hypothesis: Mean of fuel-efficiency of 2008 cars > Mean of fuel-efficiency of 1999 cars.


# In[26]:


from pydataset import data

mpg = data('mpg')
mpg = pd.DataFrame(mpg)
mpg


# In[27]:


mpg['avg_mileage'] = (mpg['cty'] + mpg['hwy']) / 2
mpg


# In[28]:


cars_08_mileage = mpg[mpg['year'] == 2008]['avg_mileage']
cars_99_mileage = mpg[mpg['year'] == 1999]['avg_mileage']

cars_08_mileage.describe()


# In[29]:


mean_08 = cars_08_mileage.mean()
sd_08 = cars_08_mileage.std()


# In[30]:


cars_99_mileage.describe()


# In[31]:


mean_99 = cars_99_mileage.mean()
sd_99 = cars_99_mileage.std()


# In[32]:


cars_08 = stats.norm(mean_08, sd_08).rvs(117)
cars_99 = stats.norm(mean_99, sd_99).rvs(117)
plt.hist(cars_08)


# In[33]:


plt.hist(cars_99)


# In[34]:


print(cars_08.var())
print(cars_99.var())


# In[40]:


t2, p2 = stats.ttest_ind(cars_08, cars_99, equal_var=False)
t2, p2


# In[42]:


alpha = .05

if p2 <= alpha:
    print('Reject null')
else:
    print('Fail to reject null')


# Takeaway: we failed to reject the null hypothesis there is no significant difference in fuel efficiency between cars made between the years 1999 and 2008

# #### Are compact cars more fuel-efficient than the average car?
#  - 1 sample t-test

# Null: compact cars are not more efficient than the average car
# 
# Alternate: copact cars are more efficient than the average car

# In[43]:


compact_mpg = mpg[mpg['class'] == 'compact']['avg_mileage']
compact_mpg.describe()


# In[46]:


comp_mean = compact_mpg.mean()
comp_sd = compact_mpg.std()


# In[47]:


avg_mile_mean = mpg['avg_mileage'].mean()


# In[48]:


comp = stats.norm(comp_mean, comp_sd).rvs(47)


# In[49]:


plt.hist(comp)


# In[51]:


t3, p3 = stats.ttest_1samp(comp, avg_mile_mean)
t3, p3


# In[53]:


alpha = .05

if t3 > 0 and p3/2 <= alpha:
    print('Reject null')
else:
    print('Fail to reject null')


# Takeaway: reject the null hypothesis, compact cars are significantly more efficient than the average car.

# #### Do manual cars get better gas mileage than automatic cars?
#  - indpendent t-test
#  - 1 tail

# Null: Manual cars are not more efficient than automatics.
# 
# Alternate: manual cars are more efficient than automatics

# In[54]:


manual_mpg = mpg[mpg['trans'].str.startswith('m')]['avg_mileage']
manual_mpg.describe()


# In[55]:


auto_mpg = mpg[mpg['trans'].str.startswith('a')]['avg_mileage']
auto_mpg.describe()


# In[56]:


manual_mean = manual_mpg.mean()
manual_sd = manual_mpg.std()


# In[57]:


auto_mean = auto_mpg.mean()
auto_sd = auto_mpg.std()


# In[58]:


manual = stats.norm(manual_mean, manual_sd).rvs(77)
auto = stats.norm(auto_mean, auto_sd).rvs(157)


# In[59]:


plt.hist(manual)


# In[60]:


plt.hist(auto)


# In[61]:


print(manual.var())
print(auto.var())


# In[62]:


t4, p4 = stats.ttest_ind(manual, auto, equal_var=False)
t4, p4


# In[64]:


alpha = .05

if t4 > 0 and p4/2 < alpha:
    print('Reject null')
else:
    print('Fail to reject null')


# Takaway: manual cars are significantly more efficient than automatic cars

# ## Exercises

# ###Answer with the type of stats test you would use (assume normal distribution):
# 1. Is there a relationship between the length of your arm and the length of your foot?
# - 1 sample t-test

# 2. Do guys and gals quit their jobs at the same rate?
# - indepdent t-test

# 3. Does the length of time of the lecture correlate with a students grade?
# - 1 sample t-test

# 1. Does tenure correlate with monthly charges?
# 

# In[4]:


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


# In[145]:


import env

db_url = f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/telco_churn'
df = pd.read_sql('SELECT * FROM customers', db_url) 
df = pd.DataFrame(df)
df.info()


# In[147]:


sns.regplot(x = 'tenure', y= 'monthly_charges', data = df, marker = '.', line_kws = {'color': 'red'})


# In[148]:


df.monthly_charges.hist()
df.tenure.hist()


# In[149]:


stats.normaltest(df.tenure, None)


# In[151]:


df.total_charges = df.total_charges.replace(' ', np.nan).astype(float)


# In[152]:


sns.regplot(df.tenure, df.total_charges, marker = '.', line_kws = {'color': 'red'})


# In[160]:


corr, p = stats.spearmanr(df.tenure, df.monthly_charges)
corr, p


# In[161]:


corr1, p1 = stats.spearmanr(df.tenure, df.total_charges)
corr1, p1
# Strongly related


# In[162]:


def rtest(sql): 
    return pd.Series(stats.spearmanr(sql.tenure, sql.monthly_charges), index = ['r', 'p'])


# In[163]:


df.groupby(['phone_service', 'internet_service_type_id']).apply(rtest)


# In[169]:


db_url = f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/employees'
df3 = pd.read_sql('''SELECT s.emp_no, s.salary, s.to_date, e.hire_date FROM salaries AS s 
                JOIN employees AS e ON e.emp_no = s.emp_no WHERE s.to_date >= NOW();''', db_url)
df3.head()


# In[170]:


df3['tenure'] = (df3.to_date - df3.hire_date).astype(str).str.split(' ', expand = True)[0].astype(int)


# In[172]:


corr, p= stats.spearmanr(df3.tenure, df3.salary)
corr, p
# Significant relationship but not very strong


# In[174]:


sns.regplot(df3.tenure, df3.salary, marker = '.', line_kws = {'color': 'red'})


# In[176]:


query2 = 'SELECT count(title), emp_no FROM titles GROUP BY emp_no'

df4 = pd.read_sql(query2, db_url)
df.head()


# In[177]:


df5 = df3.merge(df4, left_on= 'emp_no', right_on = 'emp_no', how = 'left', indicator = True)
df5


# In[178]:


corr, p = stats.spearmanr(df5.tenure, df5['count(title)'])
corr, p


# In[180]:


slp = data('sleepstudy')
slp


# In[181]:


slp_r = slp[['Days', 'Reaction']].groupby(slp.Days).Reaction.mean().reset_index()
slp_r


# In[182]:


corr, x = stats.spearmanr(slp.Days, slp.Reaction)
corr, x


# ### Answer with the type of stats test you would use (assume normal distribution):
# 
# 1. Do students get better test grades if they have a rubber duck on their desk?
# 

# - 1 sample t-test

# 2. Does smoking affect when or not someone has lung cancer?
# 

# - Chi-Square Contingency Test

# 3. Is gender independent of a person’s blood type?
# 

# - Chi-Square Contigency Test

# 4. A farming company wants to know if a new fertilizer has improved crop yield or not
# 

# - 2 Sample t-test

# 5 .Does the length of time of the lecture correlate with a students grade?
# 

# - 2 Sample t-test

# 6. Do people with dogs live in apartments more than people with cats?
# 

# - Chi-Square Contingency Test

# 7. Use the following contingency table to help answer the question of whether using a macbook and being a codeup student are independent of each other.

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from pydataset import data


# In[67]:


alpha = 0.05

def eval_results(p, alpha, group1, group2):
    '''
    this function will take in the p-value, alpha, and a name for the 2 variables 
    you are comparing (group 1 and group 2)
    '''
    if p < alpha:
        print(f'There exists some relationship between {group1} and the {group2}. (p-value: {p})')
    else:
        print(f'There is not a significant relationship between {group1} and {group2}. (p-value: {p})')


# In[62]:


data = {'Codeup Student': [1, 49], 'Not Codeup Student': [30, 20]}
students = pd.DataFrame(data)
students


# In[63]:


chi2, p, degf, expected = stats.chi2_contingency(students)


# In[66]:


print(chi2)
print(p)
print(degf)
print(expected)


# In[68]:


eval_results(p, alpha, group1='Codeup Student', group2='Not Codeup Student')


# In[73]:


mpg = data('mpg')
mpg = pd.DataFrame(mpg)
mpg.head()


# In[77]:


# Is there a relationshiip between year and cylinder
observed = pd.crosstab(mpg['year'], mpg['cyl'])
observed


# In[80]:


chi2, p2, degf, expected = stats.chi2_contingency(observed)
p2


# In[81]:


eval_results(p2, alpha, group1='year', group2='cyl')


# In[133]:


import env

db_url = f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/employees'
df = pd.read_sql('SELECT * FROM employees AS e ' 
                 'JOIN dept_emp AS de ON de.emp_no = e.emp_no AND de.to_date > CURDATE() '
                 'JOIN titles AS t ON t.emp_no = e.emp_no AND t.to_date > CURDATE()', db_url)
df.head()


# ### Is an employee's gender independent of whether an employee works in sales or marketing? (only look at current employees)

# In[134]:


df = df.drop(columns=['hire_date', 'emp_no', 'birth_date', 'from_date', 'to_date'])
df.head()


# In[135]:


# d001 - Marketing
# d007 - Sales 
df1 = df[(df['dept_no']=='d001') | (df['dept_no']=='d007')]


table = pd.crosstab(df1['dept_no'], df1['gender'])
table


# In[136]:


chi2, p3, degf, expected = stats.chi2_contingency(table)
p3


# In[137]:


eval_results(p3, alpha, group1='Sales or Marketing', group2='gender')


# ### Is an employee's gender independent of whether or not they are or have been a manager?

# In[139]:


df['is_manager'] = df['title']=='Manager'
df


# In[140]:


table1 = pd.crosstab(df['is_manager'], df['gender'])
table1


# In[141]:


chi2, p4, degf, expected = stats.chi2_contingency(table1)
p4


# In[142]:


eval_results(p3, alpha, group1='is_manager', group2='gender')


# In[ ]:





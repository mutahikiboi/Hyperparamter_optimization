#.........HPO on Bike sharing data.........

library(mlr3verse)
library(tidyverse)
task <- tsk('bike_sharing')
task$data()%>%head()%>%view()
task%>%summary()
task
task%>%autoplot()

View(task)

#bike <- data("bike_sharing", package = "mlr3data")
##bike_sharing%>%view

#bike_sharing%>%summary

#autoplot(bike)

#ggplot(data = bike_sharing, mapping = aes(x = temperature, y = count)) + geom_point()
#train the decision tree learner
#defining the learner
learner.rpart <- lrn('regr.rpart')
learner.rpart$train(task = task)
#Error: <TaskRegr:bike_sharing> has the following unsupported feature types: character

#build a robust pipeline operator for the learner
g_rob <- ppl('robustify', task = task, learner = lrn('regr.rpart'))

g_rob%>%plot()

#Hyperband Tuner requires a budget parameter (e.g., subsampling rate, number of epochs)
#we need to include a subsample pipeline operator that will be used as the budget

gl_rpart <- GraphLearner$new(
  g_rob%>>%po('subsample')%>>% learner.rpart
)


ss <- ps(
  regr.rpart.cp = p_dbl(1e-2,1, logscale = TRUE),
  regr.rpart.minisplit = p_int(1,10),
  subsample.frac = p_dbl(1e-2,1,tags = "budget")
)

#Optimizing the graph learner with the hyperband tuner using an appropriate η value
install.packages("mlr3hyperband")
library(mlr3hyperband)
library(mlr3tuning)

m_rmse <- msr("regr.rmse")
trm_n <- trm("none")
tnr_hb <- tnr("hyperband", eta= 3)

instance <- TuningInstanceBatchSingleCrit$new(
    task = task,
    learner = gl_rpart,
    resampling = rsmp("holdout"),
    measure = m_rmse,
    search_space = ss,
    terminator = trm_n
  
)

tnr_hb$optimize(instance)

#................exercise 2.......................

"the four building blocks one needs to specify in order to create a Bayesian
tuner?
  Hint: Have a look at the mlr3mbo vignette you can find here.
1) Loop Function that iterates trough the optimization loop in the three following steps:
  • updates the surrogate model
• updates the acquisition function
• optimizes the acquisition function and returns the next point to be evaluated
2) Surrogate Model. This can be any regression learner, although Gaussian Process (regr.km,
 for low dimensional numeric search spaces) and Random Forest (regr.ranger, for higher                                                                                                                                              dimensional mixed search spaces) are suggested.
3) The Acquisition Function is used to evaluate, how much benefit the evaluation of each
point in the surrogate model will have. A good default is the Expected Improvement.
4) Acquisition Function Optimizer optimizes the Acquisition Function and returns the next
point to be evaluated."

##
#install packages and libraries
library(mlr3misc)
library(mlr3mbo)
library(bbotk)

l_xgb <- lrn("regr.xgboost",
             eta = to_tune(1e-4, 1, logscale = TRUE),
             nrounds = to_tune(1, 10),
             max_depth = to_tune(1, 20))

gl_xgb <- as_learner(ppl("robustify",
                         learner = l_xgb,
                         task = task) %>>%
                       l_xgb)
gl_xgb$id <- "gl_xgb"

# loop function ----
loop_function <- bayesopt_ego

# surrogate model ----
surrogate <- srlrn(lrn("regr.ranger"))

# acquisition function ----
acq_function <- acqf("ei") #expected improvement

# acquisition optimizer ----
optim <- opt("random_search")
term <- trm("evals")
acq_optimizer <- acqo(optim, term)

# tuner ----
tnr_bo <- tnr("mbo",
              loop_function = loop_function,
              surrogate = surrogate,
              acq_function = acq_function,
              acq_optimizer = acq_optimizer)

at_bo_xgb <- AutoTuner$new(
  learner = gl_xgb,
  tuner = tnr_bo,
  terminator = trm("evals", n_evals = 20),
  measure = msr("regr.rmse"),
  resampling = rsmp("holdout")
)

at_bo_xgb$id <- "at_bo_xgb"

# Benchmarking the tuning methods Bayesian Optimization and Random Search as well as
#the untuned version of xgboost.

at_rs_xgb <- AutoTuner$new(
  learner = gl_xgb,
  resampling = rsmp("holdout"),
  terminator = trm("evals", n_evals = 20),
  tuner = tnr("random_search"),
  measure = msr("regr.rmse")
)
at_rs_xgb$id <- "at_rs_xgb"

untuned_xgb <- as_learner(ppl("robustify",
                              learner = lrn("regr.xgboost"),
                              task = task) %>>%
                            lrn("regr.xgboost"))

untuned_xgb$id <- "xgboost"

learners <- list(
  lrn("regr.featureless"),
  untuned_xgb,
  at_rs_xgb,
  at_bo_xgb
)

resampling <- rsmp("cv", folds = 3)

# you might want to use parallelizaion to speed up the benchmark
# future::plan(list("sequential", "multisession"))
bmr <- benchmark_grid(task, learners, resampling) %>%
  benchmark(store_models = TRUE)

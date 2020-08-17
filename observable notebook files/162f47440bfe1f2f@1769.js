// https://observablehq.com/d/162f47440bfe1f2f@1769
export default function define(runtime, observer) {
  const main = runtime.module();
  const fileAttachments = new Map([["wine_dataset.csv",new URL("./files/619c7e1485f4f02e9f0f856792dbc135470d5d9d4197fd48e2922e3055b1756e88e5823d619b5720865d0620fa81d33c74277e2cbf9b450866176b3aabb48d20",import.meta.url)]]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], function(md){return(
md`# Learning to predict Wine quality using TF.js and Danfo.js
Danfo.js is the best javascript package that provides fast, flexible, and expressive data structures designed to make working with "relational" or "labelled" data both easy and intuitive. It is heavily inspired by the Pandas library and provides a similar API.
### Important !!!
* The tutorial aim at introducing you to capabilities of Danfo.js and TF.js 
* Get familiar with observable nootbook as it is different from jupyter notebook via  https://observablehq.com/@observablehq/observable-for-jupyter-users
* Uncomment a cell and run it with "Shift + Enter" one at a time to avoid errors  
* Have fun !
`
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
## Lets get started
import danfo.js and TF.js
`
)});
  main.variable(observer("dfd")).define("dfd", ["require","print"], function(require,print){return(
require('danfojs@0.0.15/dist/index.min.js').catch(() => {
  window.dfd.Series.prototype.print = window.dfd.DataFrame.prototype.print = function() { return print(this) };
  return window.dfd;
})
)});
  main.variable(observer("tf")).define("tf", ["require"], function(require){return(
require('@tensorflow/tfjs')
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Data ingestion 
`
)});
  main.variable(observer("df")).define("df", ["dfd","FileAttachment"], async function(dfd,FileAttachment){return(
dfd.read_csv(await FileAttachment("wine_dataset.csv").url())
)});
  main.variable(observer()).define(["print","df"], function(print,df){return(
print(df.head())
)});
  main.variable(observer()).define(["df"], function(df){return(
df.shape
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### column renaming 
Renaming "type" feature to "wine_type"
`
)});
  main.variable(observer("df1")).define("df1", ["df"], function(df){return(
df.rename({mapper:{'type':'wine_type'}})
)});
  main.variable(observer()).define(["print","df1"], function(print,df1){return(
print(df1.head())
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Data preprocessing and EDA
`
)});
  main.variable(observer()).define(["print","df1"], function(print,df1){return(
print(df1.describe())
)});
  main.variable(observer()).define(["print","df1"], function(print,df1){return(
print(df1.isna().sum())
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Percentage of missing data 
`
)});
  main.variable(observer()).define(["print","df1"], function(print,df1){return(
print(df1.isna().sum().div(df1.isna().count()).round(4))
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### fill missing values mean  
`
)});
  main.variable(observer("df_filled")).define("df_filled", ["df1"], function(df1)
{
  let df2
  df1.columns.forEach(function(feat,i){
    if(df1[feat].dtypes[0]!="string"){
      let mean = Number(df1[feat].mean().toFixed(4))
      df2 = df1.fillna({columns:[feat],values:[mean]})
    }
    
  })   
  return df2
  }
);
  main.variable(observer()).define(["print","df_filled"], function(print,df_filled){return(
print(df_filled.head())
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Encode wine_type to numbers 
`
)});
  main.variable(observer("dum_df")).define("dum_df", ["df_filled"], function(df_filled)
{
  let df1 = df_filled.replace({"replace": "white", "with": 0, "in": ["wine_type"]}) 
  let df2 = df1.replace({"replace": "red", "with": 1, "in": ["wine_type"]})
  df2.astype({column: "wine_type", dtype: "int32"})
  return df2
}
);
  main.variable(observer()).define(["print","dum_df"], function(print,dum_df){return(
print(dum_df.head())
)});
  main.variable(observer()).define(["print","dum_df"], function(print,dum_df){return(
print(dum_df.ctypes)
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### The "div" function for plotting  
`
)});
  main.variable(observer("div")).define("div", ["html"], function(html){return(
function div(fn) {
  const d = html`<div>`;
  fn(d);
  return d;
}
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Visualizations for EDA
`
)});
  main.variable(observer()).define(["dum_df"], function(dum_df){return(
dum_df.quality.nunique()
)});
  main.variable(observer()).define(["print","dum_df"], function(print,dum_df){return(
print(dum_df.quality.value_counts())
)});
  main.variable(observer()).define(["dum_df","div"], function(dum_df,div)
{
  let layout = {
  title: 'Wine Quality Counts Plot',
}
let quality_count = dum_df.quality.value_counts()
return div(d => quality_count.plot(d).bar({layout:layout}))
}
);
  main.variable(observer()).define(["dum_df","div"], function(dum_df,div)
{
  let layout = {
  title: 'wine quality proportion',
}
let quality_count = dum_df.quality.value_counts()
return div(d => quality_count.plot(d).pie({layout:layout}))
}
);
  main.variable(observer()).define(["md"], function(md){return(
md`
### Binning the target feature
here 6 wine quality values based from the results above. Minimum is 3 and maximum is 9. We can create 3 wine quality categories namely poor quality, normal quality, excellent quality.

* if quality < 6 - poor quality
* if quality = 6 - normal quality
* if quality > 6 - excellent quality
`
)});
  main.variable(observer()).define(["dum_df"], function(dum_df)
{
let mapper = {1:0,2:0,3:0,4:0,5:0,6:1,7:2,8:2,9:2}
let new_cols = dum_df['quality'].map(mapper).values
dum_df.addColumn({ "column": "wine_quality", "value": new_cols })
}
);
  main.variable(observer()).define(["print","dum_df"], function(print,dum_df){return(
print(dum_df.head())
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### drop "quality" after binning 
`
)});
  main.variable(observer("df4")).define("df4", ["dum_df"], function(dum_df){return(
dum_df.drop({ columns: ["quality"], axis: 1, inplace: false})
)});
  main.variable(observer()).define(["print","df4"], function(print,df4){return(
print(df4.head())
)});
  main.variable(observer()).define(["df4","div"], function(df4,div)
{
  let layout = {
  title: 'newly grouped wine quality bar plot',
     xaxis: {
        title: 'wine quality',
    },
    yaxis: {
        title: 'counts',
    }
}
let quality_count = df4.wine_quality.value_counts()
return div(d => quality_count.plot(d).bar({layout:layout}))
}
);
  main.variable(observer()).define(["md"], function(md){return(
md`
### Feature distribution  
`
)});
  main.variable(observer()).define(["div","dum_df"], function(div,dum_df)
{
  let layout = {
  title: 'fixed acidity Distribution',
}
return div(d => dum_df['fixed acidity'].plot(d).hist({layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'fixed acidity Distribution',
}
return div(d => df4['volatile acidity'].plot(d).hist({layout:layout}))
}
);
  main.variable(observer()).define(["md"], function(md){return(
md`
### Feature relationship to target feature visualization 
`
)});
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'fixed acidity relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"fixed acidity",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'volatile acidity relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"volatile acidity",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'citric acid relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"citric acid",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'residual sugar relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"residual sugar",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'chlorides relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"chlorides",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'free sulfur dioxide relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"free sulfur dioxide",layout:layout}))
}
);
  main.variable(observer()).define(["div","df4"], function(div,df4)
{
  let layout = {
  title: 'total sulfur dioxide relationship to target feature',
}
return div(d => df4.plot(d).bar({x:"wine_quality",y:"total sulfur dioxide",layout:layout}))
}
);
  main.variable(observer()).define(["print","df4"], function(print,df4){return(
print(df4.head())
)});
  main.variable(observer()).define(["df4","div"], function(df4,div)
{
  let sub_df = df4.loc({ columns: ["fixed acidity","volatile acidity", "citric acid","residual sugar","chlorides",
                                   "free sulfur dioxide","total sulfur dioxide","density","pH","alcohol"] })
  let layout = {
  title: 'features box plot before scaling', xaxis: {title: 'X',},yaxis: {title: 'Y',
    }
}
return div(d => sub_df.plot(d).box({layout:layout}))
}
);
  main.variable(observer()).define(["md"], function(md){return(
md`
### Data Scaling
`
)});
  main.variable(observer("df_scaled")).define("df_scaled", ["dfd","df4"], function(dfd,df4)
{
let scaler = new dfd.MinMaxScaler()
let X = df4.iloc({ columns: ["0:11"] })
scaler.fit(X)
let scaled_df =  scaler.transform(X)
return new dfd.DataFrame(scaled_df.values,{columns: df4.columns.slice(0,12)}).round(4)
}
);
  main.variable(observer()).define(["print","df_scaled"], function(print,df_scaled){return(
print(df_scaled.tail())
)});
  main.variable(observer()).define(["md"], function(md){return(
md` #### Add target feature to the dataframe`
)});
  main.variable(observer()).define(["df_scaled","df4"], function(df_scaled,df4){return(
df_scaled.addColumn({ "column": "wine_quality", "value": df4.wine_quality.values })
)});
  main.variable(observer()).define(["print","df_scaled"], function(print,df_scaled){return(
print(df_scaled.head())
)});
  main.variable(observer()).define(["df_scaled","div"], function(df_scaled,div)
{
  let sub_df = df_scaled.loc({ columns: ["fixed acidity","volatile acidity", "citric acid","residual sugar","chlorides",
                                   "free sulfur dioxide","total sulfur dioxide","density","pH","alcohol"] })
  let layout = {
  title: 'box plot of scaled dataset', xaxis: {title: 'X',},yaxis: {title: 'Y',
    }
}
return div(d => sub_df.plot(d).box({layout:layout}))
}
);
  main.variable(observer()).define(["md"], function(md){return(
md`
# OneHotEncoding of the target feature since its a multiclass
`
)});
  main.variable(observer("y_hot")).define("y_hot", ["dfd","df_scaled"], function(dfd,df_scaled)
{
  let encode = new dfd.OneHotEncoder()
  let y = df_scaled["wine_quality"].values
  encode.fit(y)
  let y_enc = encode.transform(y)
  return y_enc
}
);
  main.variable(observer()).define(["print","y_hot"], function(print,y_hot){return(
print(y_hot)
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### merge X with encoded y
`
)});
  main.variable(observer("df5")).define("df5", ["df_scaled","dfd","y_hot"], function(df_scaled,dfd,y_hot)
{
  let X = df_scaled.drop({ columns: ["wine_quality"], axis: 1, inplace: false})
  let com_df = dfd.concat({ df_list: [X, y_hot], axis: 1 })
  return com_df
}
);
  main.variable(observer()).define(["print","df5"], function(print,df5){return(
print(df5.head())
)});
  main.variable(observer("n_rows")).define("n_rows", ["df_scaled"], function(df_scaled){return(
df_scaled.shape[0]
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Train Test splitting for model training and evaluation 
`
)});
  main.variable(observer("train_test_split")).define("train_test_split", ["df_scaled","df5"], function(df_scaled,df5)
{ let df_len = df_scaled.shape[0]
  let df_sample = df5.sample(df_len)
  let X = df5.iloc({columns: ["0:11"]})
  let y = df5.iloc({columns: ["12:"]})
  let X_train = X.iloc({rows: ["0:4999"]}).tensor
  let y_train = y.iloc({rows: ["0:4999"]}).tensor
  let X_test = X.iloc({rows: ["4999:"]}).tensor
  let y_test = y.iloc({rows: ["4999:"]}).tensor
  let data_array = [X_train,X_test,y_train,y_test]
  return data_array
  }
);
  main.variable(observer()).define(["train_test_split"], function(train_test_split){return(
train_test_split[1].shape
)});
  main.variable(observer("input_shape")).define("input_shape", ["train_test_split"], function(train_test_split){return(
train_test_split[0].shape[1]
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### Instatiate MLP model with TF.js
`
)});
  main.variable(observer("model")).define("model", ["tf","input_shape"], function(tf,input_shape)
{

  // Define a model for linear regression.
  const model = tf.sequential({
    layers: [
      tf.layers.dense({inputShape: [input_shape], units: 32, activation: 'relu'}),
      tf.layers.dense({units: 64, activation: 'relu'}),
      tf.layers.dense({units: 128, activation: 'relu'}),
      tf.layers.dense({units: 64, activation: 'relu'}),
      tf.layers.dense({units: 32, activation: 'relu'}),
      tf.layers.dense({units: 3, activation: 'softmax'}),
    ]
  });

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({
    optimizer: 'sgd',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  return model
}
);
  main.variable(observer("validation_acc")).define("validation_acc", function(){return(
[]
)});
  main.variable(observer("training_acc")).define("training_acc", function(){return(
[]
)});
  main.variable(observer()).define(["md"], function(md){return(
md`
### fit model on X_train and y_train with 20% validation split
`
)});
  main.variable(observer()).define(["model","train_test_split","training_acc","validation_acc","tf"], function(model,train_test_split,training_acc,validation_acc,tf){return(
model.fit(train_test_split[0], train_test_split[2], {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
 
  callbacks: {onEpochEnd: (batch, logs) => {
       
    training_acc.push(logs.acc)
    validation_acc.push(logs.val_acc)
    
  }, callbacks :tf.callbacks.earlyStopping({monitor: 'val_acc'})}
 }).then(info => {
  
 })
)});
  main.variable(observer()).define(["md"], function(md){return(
md` ### Create dataframe for evaluation
`
)});
  main.variable(observer("df_eva")).define("df_eva", ["training_acc","validation_acc","dfd"], function(training_acc,validation_acc,dfd)
{
  let data = {"train_acc":training_acc,"val_acc":validation_acc}
  let df_output = new dfd.DataFrame(data)
  return df_output
}
);
  main.variable(observer()).define(["md"], function(md){return(
md` 
### Model evaluation
`
)});
  main.variable(observer()).define(["div","df_eva"], function(div,df_eva)
{
  let layout = {
  title: 'Model Evaluation',
    xaxis: {
        title: 'Number of Epochs',
    },
    yaxis: {
        title: 'Accuracy',
    }
}
return div(d => df_eva.plot(d).line({layout:layout}))
}
);
  main.variable(observer("print")).define("print", ["html"], function(html){return(
function print(df) {
  const {col_types, series, columns, index, values} = df;
  const table = html`
    <div style="overflow: auto; max-height: 300px;">
    <table class="df-table">
      <thead>
        <tr>
          <th></th>
          ${series
            ? html`<th class="${col_types[0]}">${columns}</th>`
            : columns.map((name, i) => html`<th class="${col_types[i]}">${name}</th>`)}
        </tr>
      </thead>
      <tbody>
        ${values.map((row, i) => html`
          <tr>
            <th>${index[i]}</th>
            ${series
              ? html`<td class="${col_types[0]}">${row}</td>`
              : row.map((v, j) => html`<td class="${col_types[j]}">${v}</td>`)}
          </tr>
        `)}
      </tbody>
    </table>
    </div>
    <style>
      table.df-table { white-space: pre; }
      table.df-table th, td { padding: 2px 5px; font-variant-numeric: tabular-nums; }
      table.df-table .float32, .int32 { text-align: right; }
    </style>
  `;
  table.value = df;
  return table;
}
)});
  main.variable(observer()).define(["md"], function(md){return(
md`# References 
 
* Code for danfo.js inspiration from : https://observablehq.com/@visnup/hello-danfo-js
* Code for TF.js inspiration from: https://www.tensorflow.org/js/guide/train_models
`
)});
  return main;
}

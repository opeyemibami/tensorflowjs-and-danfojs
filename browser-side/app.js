// import the packages
const dfd = require("danfojs-node")
const tf = require("@tensorflow/tfjs-node")

// Loading and processing your data
async function load_process_data() {
    
    let df = await dfd.read_csv("https://github.com/opeyemibami/tensorflowjs-and-danfojs/blob/master/dataset/wine_dataset.csv")
    df.head().print() 
}

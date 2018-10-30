//# ******************************************************************************
//# Copyright 2017-2018 Intel Corporation
//#
//# Licensed under the Apache License, Version 2.0 (the "License");
//# you may not use this file except in compliance with the License.
//# You may obtain a copy of the License at
//#
//#     http://www.apache.org/licenses/LICENSE-2.0
//#
//# Unless required by applicable law or agreed to in writing, software
//# distributed under the License is distributed on an "AS IS" BASIS,
//# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//# See the License for the specific language governing permissions and
//# limitations under the License.
//# ******************************************************************************

const api = '/inference';
const models = [
    { name: 'jieba_pos', description: '词法类API'},
    { name: 'ner', description: '实体识别API'},
    //{ name: 'spacy_ner', description: 'Name Entity Recognition using spacy',},
    //{ name: 'bist', description: 'BIST Dependency Parser'},
    { name: 'gen', description: '自动生成文本API'},
    { name: 'sentiment_classify', description: '情感分析API'},
    { name: 'word2vec', description: '测试方法'}
];

let annotateForm = document.getElementById('annotateForm');
let sentenceInput = annotateForm.sentence;
let annotateBtn = annotateForm.annotateBtn;
let dropdown = annotateForm.dropdown;

annotateForm.onsubmit = (event) => {
    event.preventDefault();
    doAnnotate(sentenceInput.value);
}

function populateDropdown() {
    models.forEach(model => {
        let option = document.createElement('option');
        option.text = model.description;
        dropdown.add(option);
    });
}

function doAnnotate(text) {
    let displacyParseDiv = document.getElementById("displacy_parse");
    let displacyDiv = document.getElementById("displacy");
    while (displacyDiv.firstChild) {
        displacyDiv.removeChild(displacyDiv.firstChild);
    }
    while(displacyParseDiv.firstChild) {
        displacyParseDiv.removeChild(displacyParseDiv.firstChild);
    }
    postData(text, api);
}

function getRandomColor() {
    let o = Math.round, r = Math.random, s = 255;
    let r_1 = o(r() * s);
    let r_2 = o(r() * s);
    let r_3 = o(r() * s);
    let rgb = 'rgb(' + r_1 + ',' + r_2 + ',' + r_3 + ')';
    let rgba = 'rgba(' + r_1 + ',' + r_2 + ',' + r_3 + ', 0.2)';
    return [rgb, rgba];
}

function addColorToAnnotationSet(annotation_set) {
    dataLength = annotation_set.length;
    for (let i = 0; i < dataLength; i++) {
        let colors = getRandomColor();
        let rgb_color = colors[0];
        let rgba_color = colors[1];
        let data1 = '[data-entity][data-entity=' + annotation_set[i] + '] {background: ' + rgba_color + '; border-color: ' + rgb_color + ';}';
        let data2 = '[data-entity][data-entity=' + annotation_set[i] + ']::after {background: ' + rgb_color + ';}';
        document.styleSheets[0].insertRule(data1);
        document.styleSheets[0].insertRule(data2);
    }
}

function renderData(data, type) {
    if (type == 'core') {
        //    for dependency
        let dataLength = data.length;
        for (let i = 0; i < dataLength; i++) {
            let displacy_div = document.getElementById("displacy_parse");
            let child_node = document.createElement("div");        // Create a child div node
            let div_id = "displacy_div" + i;
            child_node.setAttribute("id", div_id);  // set id
            displacy_div.appendChild(child_node);  // add to html the new div
            displacy_div.appendChild(document.createElement("br"));

            let displacy_core = new displaCy(api, {
                container: '#' + div_id,
                format: 'spacy',
                distance: 150,
                offsetX: 100,
                collapsePunct: false,
                collapsePhrase: false,
                bg: '#003c72',
                color: 'white',
                wordSpacing: 50
            });
            displacy_core.render(data[i]);
        }
    }
    else if (type == 'high_level') {
        let displacy_core = new displaCyENT(api, {
            container: '#displacy',
            distance: 150,
            offsetX: 100,
            collapsePunct: false,
            collapsePhrase: false,
            bg: '#006680',
            color: 'white',
            wordSpacing: 50
        });
        addColorToAnnotationSet(data.annotation_set);
        displacy_core.render(data.doc_text, data.spans, data.annotation_set);
    }
    else if (type == 'chart') {
        let chart = new Chart(ctx, {
        // The type of chart we want to create
        type: 'line',
        // The data for our dataset
        data: {
            labels: [1,2,3,4,5,6,7],
            datasets: [{
                backgroundColor: 'rgb(255, 255, 255)',//绘制双曲线的时候最好使用rgba,要不不透明的白色背景可以使得某些线条绘制不出来
                borderColor: 'rgb(255, 99, 132)',
                data: [0, 10, 5, 2, 20, 30, 45],
            }]
        },
        // 配置文件
        options: {
            //标题设置
            title:{
                display:true,
                text:'每圈速度',
            },
            //禁用动画
            animation:{
              duration:0,
            },
            hover:{
                animationDuration:0,
            },
            responsiveAnimationDuration: 0,
            //提示功能
            tooltips:{
              enable:false
            },
            //顶部的文字提示
            legend:{
              display:false,
            },
            //设置x,y轴网格线显示,标题等
            scales :{
                xAxes:[{
                    //轴标题
                    scaleLabel:{
                        display:true,
                        labelString:'第几圈',
                        fontColor:'#666'
                    },
                    //网格显示
                    gridLines:{
                        display:false
                    },


                }],
                yAxes:[{
                    scaleLabel:{
                        display:true,
                        labelString:'成绩/s'
                    },
                    gridLines:{
                        display:false
                    },

                }],

            },

            //禁用赛尔曲线
            elements: {
                line: {
                    tension: 0,
                }
            }
        }
    });
    }

    else {
        displayError("Error, bad response from server - no service type selected");
    }
}

function postData(text, url) {
    let selectedModel = models[dropdown.selectedIndex].name;
    fetch(url, {
        body: JSON.stringify({ model_name: selectedModel, "docs": [{ "id": 1, "doc": text }] }),
        cache: 'no-cache',
        credentials: 'same-origin',
        headers: {
            'Access-Control-Allow-Origin': '*',
            'content-type': 'application/json',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Methods': '*',
            'Response-Format': 'json',
            'IS-HTML': 'True'
        },
        method: 'POST',
        mode: 'cors',
        redirect: 'follow',
        referrer: 'no-referrer',
    })
    .then(response => response.json())
    .then(data => renderData(data[0].doc, data[0].type))
    .catch(error => displayError(error))
}
function displayError(error) {
    let errorAlert = document.getElementById('error_alert');
    errorAlert.innerHTML = error;
    if (errorAlert.classList.contains('d-none')) {
        errorAlert.classList.remove('d-none');
    }
}
populateDropdown();
# Machine Learning Application for Trip Destination Component of Activity Based Model
Ryan Liao, Minjin Li, Emily Chen


### Introduction
We partnered with San Diego Association of Governments (SANDAG) to understand the role of activity based models (ABMs) in San Diego’s regional transportation plan. Recent work shows the implementation of ABMs becoming a more common practice in regional transportation planning for large, urban U.S. regions. ABMs are computational models that simulate individual and household decisions and activity patterns that comprise their everyday travel. ABMs assist with understanding the effect of different policies on travel behavior.

Prior work in ABMs utilize statistical models, which are computationally slow.
The SANDAG ABM involves a costly data collection process and requires 40 minutes for trip destination prediction. This project aims to improve the efficiency of SANDAG's ABM Trip Destination component with machine learning. 

### Data

### Methods

### Results

### Discussion & Conclusion

<div class="tab-container">
    <button class="tablink" onclick="toggleMenu('menu1')">Menu 1</button>
    <button class="tablink" onclick="toggleMenu('menu2')">Menu 2</button>
    <button class="tablink" onclick="toggleMenu('menu3')">Menu 3</button>
</div>

<div id="menu1" class="menu">
    <h3>Menu 1 Content</h3>
    <p>This is the content of Menu 1.</p>
</div>

<div id="menu2" class="menu">
    <h3>Menu 2 Content</h3>
    <p>This is the content of Menu 2.</p>
</div>

<div id="menu3" class="menu">
    <h3>Menu 3 Content</h3>
    <p>This is the content of Menu 3.</p>
</div>

<script>
function toggleMenu(menuId) {
    var menu = document.getElementById(menuId);
    if (menu.style.display === "block") {
        menu.style.display = "none";
    } else {
        menu.style.display = "block";
    }
}
</script>

export let MENU_ITEM = [
    {
        path: 'index',
        title: 'Home',
        icon: 'home'
    },
    {
        path: 'form/form-layouts',
        title: 'Live Tweet Search',
        icon: 'search'
    },
    {
        path: 'preprocess',
        title: 'Tweet Preprocessing',
        icon: 'search'
    },
    {
        path: 'sentiment',
        title: 'Sentiment Analysis',
        icon: 'search'
    },
    {
        path: 'visuals',
        title: 'Visualisation',
        icon: 'bar-chart'
    },
    {
        path: 'form',
        title: 'Forms',
        icon: 'check-square-o',
        children: [
            {
                path: 'form-inputs',
                title: 'Form Inputs'
            },
            {
                path: 'form-layouts',
                title: 'Form Layouts'
            },
            {
                path: 'file-upload',
                title: 'File Upload'
            },
            {
                path: 'ng2-select',
                title: 'Ng2-Select'
            }
        ]
    },
    {
        path: 'charts',
        title: 'Charts',
        icon: 'bar-chart',
        children: [
            {
                path: 'echarts',
                title: 'Echarts'
            }
        ]
    },
];

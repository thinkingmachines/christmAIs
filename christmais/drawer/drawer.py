# -*- coding: utf-8 -*-

"""Create 3x3 Quick, Draw! Images from a given text"""


# Import standard library
import os
import glob
import pathlib
import logging

# Import modules
import coloredlogs
from selenium import webdriver

HTML_TEMPLATE = """
<html>
<head>
  <meta charset="UTF-8">
  <script language="javascript" type="text/javascript" src="https://storage.googleapis.com/tm-christmais/models/{model}.js"></script>
  <script language="javascript" type="text/javascript" src="{p5_path}"></script>
  <script language="javascript" type="text/javascript" src="{p5_svg_path}"></script>
  <script language="javascript" type="text/javascript" src="{numjs_path}"></script>
  <script language="javascript" type="text/javascript" src="{sketch_rnn_path}"></script>
  <script language="javascript" type="text/javascript" src="{generate_path}"></script>
  <style> body {{padding: 0; margin: 0;}} </style>
</head>
<body>
  <div id="sketch"></div>
</body>
</html>
"""


class Drawer:
    """Draw a 3x3 image form a given string"""

    def __init__(self, webdriver_path='./webdriver/chromedriver'):
        """Initialize the class

        Parameters
        ----------
        webdriver_path : str (default is ./webdriver/chromedriver)
            path to chromedriver
        """
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(logging.INFO, logger=self.logger)
        self.logger.debug('Creating chromedriver...')
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        self.driver = webdriver.Chrome(webdriver_path, chrome_options=options)

    def draw(self, label, outfile):
        """Generate an image based on the output label

        Parameters
        ----------
        label : str
            Quick, Draw! class
        outfile : str
            Path to save the resulting image into
        """
        # create an html file first
        self._create_index_html(label)
        self._draw_png(outfile)

    def _read_categories(self, category='categories.txt'):
        """Read category files

        Parameters
        ----------
        category : str
            Name of categories file (default is categories.txt)
        """
        with open(
            glob.glob('./**/{}'.format(category), recursive=True)[0], 'r'
        ) as fp:
            data = fp.readlines()
        return [d.rstrip() for d in data]

    def _create_index_html(self, label):
        """Generate an HTML file with all the necessary packages loaded

        Parameters
        ----------
        label : str
            Quick, Draw! class
        """
        p5_path = glob.glob('./**/p5.js', recursive=True)
        p5_svg_path = glob.glob('./**/p5.svg.js', recursive=True)
        numjs_path = glob.glob('./**/numjs.js', recursive=True)
        sketch_rnn_path = glob.glob('./**/sketch_rnn.js', recursive=True)
        generate_path = glob.glob('./**/generate.js', recursive=True)

        categories = self._read_categories()
        if label in categories:
            with open('index.html', 'w') as fp:
                fp.write(
                    HTML_TEMPLATE.format(
                        model=label,
                        p5_path=p5_path[0],
                        p5_svg_path=p5_svg_path[0],
                        numjs_path=numjs_path[0],
                        sketch_rnn_path=sketch_rnn_path[0],
                        generate_path=generate_path[0],
                    )
                )
                self.logger.debug('Created index.html file!')
        else:
            msg = 'Missing {} in categories.txt!'
            self.logger.error(msg.format(label))
            raise ValueError(msg.format(label))

    def _draw_png(self, outfile):
        """Draw a 3x3 PNG file for the Quick, Draw! drawings

        Parameters
        ----------
        outfile : str
            Path to save the resulting image into
        """
        index_path = os.path.abspath('index.html')
        index_uri = pathlib.Path(index_path).as_uri()
        self.logger.debug('Opening index.html...')
        self.driver.get(index_uri)
        svg = self.driver.find_element_by_tag_name('svg')
        svg.screenshot(outfile)
        self.logger.debug('Screenshot saved at {}'.format(outfile))

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
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(webdriver_path, chrome_options=options)
        self.index_folder = './generated_html/'
        self.png_folder = './generated_png/'
        # Create artifacts folder
        if not os.path.exists(self.index_folder):
            self.logger.debug('Creating html directory...')
            os.makedirs(self.index_folder)
        if not os.path.exists(self.png_folder):
            self.logger.debug('Creating png directory...')
            os.makedirs(self.png_folder)

    def draw(self, label, outfile):
        """Generate an image based on the output label

        Parameters
        ----------
        label : str
            Quick, Draw! class
        outfile : str
            Path to save the resulting image into
        """
        # Create an html file first: (outfile).html
        self._create_index_html(outfile=outfile, label=label)
        # Then draw a 3x3 image from (outfile).html: (outfile).png
        self._draw_png(outfile=outfile)

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

    def _create_index_html(self, outfile, label):
        """Generate an HTML file with all the necessary packages loaded

        Parameters
        ----------
        outfile: str
            Filename to save index.html into (no file extension)
        label : str
            Quick, Draw! class
        """
        source_uri = 'https://storage.googleapis.com/tm-christmais/cdn/'
        p5_path = source_uri + 'p5.js'
        p5_svg_path = source_uri + 'p5.svg.js'
        numjs_path = source_uri + 'numjs.js'
        sketch_rnn_path = source_uri + 'sketch_rnn.js'
        generate_path = source_uri + 'generate.js'

        categories = self._read_categories()
        # Reverse quickdraw names
        reverse_qd_names = {
            "alarm_clock": "clock",
            "diving_board": "board",
            "cruise_ship": "ship",
            "fire_hydrant": "hydrant",
            "palm_tree": "tree",
            "power_outlet": "outlet",
            "the_mona_lisa": "painting",
        }
        try:
            label_ = reverse_qd_names[label]
        except KeyError:
            label_ = label
        if label_ in categories:
            with open(self.index_folder + outfile + '.html', 'w') as fp:
                fp.write(
                    HTML_TEMPLATE.format(
                        model=label,
                        p5_path=p5_path,
                        p5_svg_path=p5_svg_path,
                        numjs_path=numjs_path,
                        sketch_rnn_path=sketch_rnn_path,
                        generate_path=generate_path,
                    )
                )
                self.logger.debug(
                    'Created index file!: {}.html'.format(outfile)
                )
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
        index_path = os.path.abspath(self.index_folder + outfile + '.html')
        index_uri = pathlib.Path(index_path).as_uri()
        self.logger.debug('Opening index file: {}.html'.format(outfile))
        self.driver.get(index_uri)
        svg = self.driver.find_element_by_tag_name('svg')
        svg.screenshot(self.png_folder + outfile + '.png')
        self.logger.debug('Screenshot saved at {}.png'.format(outfile))

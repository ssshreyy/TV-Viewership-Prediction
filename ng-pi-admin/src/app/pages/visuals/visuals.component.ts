import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-visual',
  templateUrl: './visuals.component.html',
  styleUrls: ['./visuals.component.scss']
})
export class VisualsComponent implements OnInit {

  serverData: JSON;
  arrayOfKeys;

  // constructor() { }
  constructor(private httpClient: HttpClient) {
  }

  ngOnInit() {
  }

  wordcloud(data){
    console.log(data);

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type':  'application/x-www-form-urlencoded',
        'Authorization': 'my-auth-token'
      })
    };
    var body = "show=" + data.show + "&year=" + data.year;

    this.httpClient.post("http://127.0.0.1:5003/search", body, httpOptions).subscribe((data) => {
      this.serverData = data as JSON;
      // alert(JSON.stringify(this.serverData))
      console.log(this.serverData)
      // alert(typeof this.serverData)
      this.arrayOfKeys = Object.keys(this.serverData)
      // alert(this.arrayOfKeys)
    })
  }
}
